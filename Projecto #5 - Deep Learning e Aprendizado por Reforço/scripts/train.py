from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import shutil
from ntxuva.rl.trainer import SelfPlayDQNTrainer, TrainConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Treino DQN por self-play para Ntxuva")
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/training"))
    parser.add_argument("--capture-reward", type=float, default=0.0)
    args = parser.parse_args()

    cfg = TrainConfig(
        episodes=args.episodes,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        capture_reward_scale=args.capture_reward,
    )
    trainer = SelfPlayDQNTrainer(cfg)
    trainer.train(args.output)

    final = args.output / "models" / "ntxuva_dqn_final.pt"
    best = args.output / "models" / "ntxuva_dqn_best.pt"
    source = best if best.exists() else final
    target = Path("models/ntxuva_dqn_best.pt")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    print(f"Modelo disponível para a GUI em: {target}")


if __name__ == "__main__":
    main()
