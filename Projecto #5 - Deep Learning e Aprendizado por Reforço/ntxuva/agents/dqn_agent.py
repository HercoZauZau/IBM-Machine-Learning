from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ntxuva.game.game import NtxuvaGame
from ntxuva.game.move import Move
from ntxuva.rl.network import QNetwork


class DQNAgent:
    name = "DQN"

    def __init__(
        self,
        state_size: int,
        action_size: int,
        model_path: str | Path | None = None,
        hidden_size: int = 256,
        device: str | None = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.state_size = state_size
        self.action_size = action_size

        payload = None
        if model_path is not None:
            payload = torch.load(Path(model_path), map_location=self.device, weights_only=False)
            state_dict = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
            # Infere a largura da primeira camada para suportar checkpoints com
            # configurações diferentes da predefinição.
            first_weight = state_dict.get("net.0.weight")
            if first_weight is not None:
                hidden_size = int(first_weight.shape[0])

        self.network = QNetwork(state_size, action_size, hidden_size=hidden_size).to(self.device)
        self.network.eval()
        self.metadata: dict = {}
        if payload is not None:
            self._load_payload(payload)

    @torch.no_grad()
    def select_action(self, game: NtxuvaGame, epsilon: float = 0.0) -> int:
        legal = game.legal_action_ids()
        if not legal:
            raise RuntimeError("Sem jogadas válidas")
        if epsilon > 0 and np.random.random() < epsilon:
            return int(np.random.choice(legal))

        state = torch.tensor(game.encode_state(), dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.network(state).squeeze(0)
        mask = torch.full_like(q, float("-inf"))
        mask[torch.tensor(legal, dtype=torch.long, device=self.device)] = 0.0
        action = int(torch.argmax(q + mask).item())
        return action

    def select_move(self, game: NtxuvaGame, epsilon: float = 0.0) -> Move:
        return game.codec.decode(self.select_action(game, epsilon=epsilon))

    def _load_payload(self, payload) -> None:
        state_dict = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
        self.network.load_state_dict(state_dict)
        if isinstance(payload, dict):
            self.metadata = {k: v for k, v in payload.items() if k != "model_state_dict"}
        self.network.eval()

    def load(self, path: str | Path) -> None:
        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        self._load_payload(payload)

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.network.state_dict(),
                "metadata": metadata or {},
            },
            path,
        )
