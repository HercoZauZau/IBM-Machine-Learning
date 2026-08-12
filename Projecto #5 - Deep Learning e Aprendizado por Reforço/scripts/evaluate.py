from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from ntxuva.agents.dqn_agent import DQNAgent
from ntxuva.agents.minimax_agent import MinimaxAgent
from ntxuva.agents.random_agent import RandomAgent
from ntxuva.game.game import NtxuvaGame
from ntxuva.rl.evaluation import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliar o agente DQN")
    parser.add_argument("--model", type=Path, default=Path("models/ntxuva_dqn_best.pt"))
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--opponent", choices=("random", "minimax"), default="random")
    parser.add_argument("--depth", type=int, default=2)
    args = parser.parse_args()

    game = NtxuvaGame()
    agent = DQNAgent(game.state_size, game.codec.action_size, model_path=args.model)
    if args.opponent == "random":
        factory = lambda: RandomAgent()
    else:
        factory = lambda: MinimaxAgent(depth=args.depth, max_children=28)

    result = evaluate(agent, factory, games=args.games)
    print(f"Jogos: {result.games}")
    print(f"Vitórias: {result.wins}")
    print(f"Derrotas: {result.losses}")
    print(f"Empates: {result.draws}")
    print(f"Win rate: {result.win_rate:.1%}")
    print(f"Jogadas médias: {result.avg_moves:.1f}")
    print(f"Capturas médias: {result.avg_captured_for:.1f} x {result.avg_captured_against:.1f}")


if __name__ == "__main__":
    main()
