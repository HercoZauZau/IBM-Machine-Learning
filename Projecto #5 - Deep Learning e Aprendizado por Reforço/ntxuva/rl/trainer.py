from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ntxuva.game.game import NtxuvaGame
from ntxuva.rl.network import QNetwork
from ntxuva.rl.replay_buffer import ReplayBuffer, Transition


@dataclass
class TrainConfig:
    episodes: int = 10_000
    gamma: float = 0.99
    learning_rate: float = 1e-4
    batch_size: int = 128
    replay_capacity: int = 100_000
    warmup_steps: int = 1_000
    train_every: int = 4
    target_update_every: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100_000
    hidden_size: int = 256
    max_moves_per_game: int = 400
    seed: int = 42
    capture_reward_scale: float = 0.0


class SelfPlayDQNTrainer:
    """DQN zero-sum com rede partilhada e self-play.

    Como o próximo estado pertence ao adversário, o alvo usa o sinal negativo:
        target = r - gamma * max Q_oponente(s', a')
    Isto aproxima a actualização minimax para jogos alternados de soma zero.
    """

    def __init__(self, config: TrainConfig, columns: int = 8, device: str | None = None):
        self.cfg = config
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        self.game = NtxuvaGame(columns=columns)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.policy = QNetwork(self.game.state_size, self.game.codec.action_size, config.hidden_size).to(self.device)
        self.target = QNetwork(self.game.state_size, self.game.codec.action_size, config.hidden_size).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay = ReplayBuffer(config.replay_capacity, seed=config.seed)
        self.global_step = 0

    def epsilon(self) -> float:
        frac = min(1.0, self.global_step / max(1, self.cfg.epsilon_decay_steps))
        return self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    @torch.no_grad()
    def choose_action(self, game: NtxuvaGame) -> int:
        legal = game.legal_action_ids()
        if np.random.random() < self.epsilon():
            return int(np.random.choice(legal))
        state = torch.tensor(game.encode_state(), dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.policy(state).squeeze(0)
        legal_t = torch.tensor(legal, dtype=torch.long, device=self.device)
        best_local = int(torch.argmax(q[legal_t]).item())
        return int(legal_t[best_local].item())

    def optimise(self) -> float | None:
        if len(self.replay) < max(self.cfg.batch_size, self.cfg.warmup_steps):
            return None
        batch = self.replay.sample(self.cfg.batch_size)
        states = torch.tensor(np.stack([t.state for t in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.stack([t.next_state for t in batch]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.bool, device=self.device)
        next_masks = torch.tensor(np.stack([t.next_mask for t in batch]), dtype=torch.bool, device=self.device)

        q_sa = self.policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target(next_states)
            next_q = next_q.masked_fill(~next_masks, float("-inf"))
            next_max = next_q.max(dim=1).values
            next_max = torch.where(torch.isfinite(next_max), next_max, torch.zeros_like(next_max))
            target = rewards - self.cfg.gamma * next_max * (~dones).float()

        loss = self.loss_fn(q_sa, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0)
        self.optimizer.step()
        return float(loss.item())

    def train(self, output_dir: str | Path) -> list[dict]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        models = output / "models"
        models.mkdir(exist_ok=True)
        history: list[dict] = []
        best_rolling = -math.inf
        rolling_results: list[float] = []

        for episode in range(1, self.cfg.episodes + 1):
            game = NtxuvaGame(columns=self.game.columns)
            total_loss = []
            episode_reward = [0.0, 0.0]
            final_result = 0.0

            for _ply in range(self.cfg.max_moves_per_game):
                if game.terminal:
                    break
                acting_player = game.current_player
                state = game.encode_state(acting_player)
                action = self.choose_action(game)
                result = game.apply_action(action)

                shaped = self.cfg.capture_reward_scale * (result.captured / 64.0)
                reward = shaped
                if result.terminal:
                    if result.winner == acting_player:
                        reward += 1.0
                    elif result.winner is None:
                        reward += 0.0
                    else:
                        reward -= 1.0

                episode_reward[acting_player] += reward
                next_state = game.encode_state(game.current_player)
                next_mask = game.action_mask() if not game.terminal else np.zeros(game.codec.action_size, dtype=np.bool_)
                self.replay.push(
                    Transition(
                        state=state,
                        action=action,
                        reward=float(reward),
                        next_state=next_state,
                        done=game.terminal,
                        next_mask=next_mask,
                    )
                )

                self.global_step += 1
                if self.global_step % self.cfg.train_every == 0:
                    loss = self.optimise()
                    if loss is not None:
                        total_loss.append(loss)
                if self.global_step % self.cfg.target_update_every == 0:
                    self.target.load_state_dict(self.policy.state_dict())

            # Limite de jogadas: empate operacional.
            if game.winner == 0:
                final_result = 1.0
            elif game.winner == 1:
                final_result = -1.0
            else:
                final_result = 0.0
            rolling_results.append(final_result)
            if len(rolling_results) > 200:
                rolling_results.pop(0)
            rolling_mean = float(np.mean(rolling_results))

            row = {
                "episode": episode,
                "moves": game.move_count,
                "winner": -1 if game.winner is None else game.winner,
                "captured_p0": int(game.captured[0]),
                "captured_p1": int(game.captured[1]),
                "epsilon": self.epsilon(),
                "mean_loss": float(np.mean(total_loss)) if total_loss else float("nan"),
                "rolling_result_p0": rolling_mean,
            }
            history.append(row)

            if episode % 100 == 0:
                print(
                    f"Ep {episode:6d}/{self.cfg.episodes} | eps={row['epsilon']:.3f} | "
                    f"moves={game.move_count:3d} | cap={game.captured.tolist()} | "
                    f"loss={row['mean_loss']:.4f}"
                )

            # Checkpoint periódico.
            if episode % 1000 == 0 or episode == self.cfg.episodes:
                self.save(models / f"ntxuva_dqn_ep{episode}.pt", episode)

            # "best" é apenas um checkpoint estável de self-play, não uma medida
            # absoluta de força; a avaliação externa é feita por evaluate.py.
            if len(rolling_results) >= 100 and rolling_mean > best_rolling:
                best_rolling = rolling_mean
                self.save(models / "ntxuva_dqn_best.pt", episode)

        self._write_history(history, output / "training_history.csv")
        self.save(models / "ntxuva_dqn_final.pt", self.cfg.episodes)
        return history

    def save(self, path: Path, episode: int) -> None:
        torch.save(
            {
                "model_state_dict": self.policy.state_dict(),
                "episode": episode,
                "config": self.cfg.__dict__,
                "state_size": self.game.state_size,
                "action_size": self.game.codec.action_size,
            },
            path,
        )

    @staticmethod
    def _write_history(history: list[dict], path: Path) -> None:
        if not history:
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
