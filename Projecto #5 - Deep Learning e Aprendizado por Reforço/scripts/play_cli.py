from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ntxuva.agents.minimax_agent import MinimaxAgent
from ntxuva.game.game import NtxuvaGame


def print_board(game: NtxuvaGame) -> None:
    print("\n", game.board)
    print("Capturas:", game.captured.tolist(), "| Vez:", game.current_player)


def main() -> None:
    game = NtxuvaGame()
    agent = MinimaxAgent(depth=2)
    while not game.terminal:
        print_board(game)
        if game.current_player == 0:
            moves = game.legal_moves()
            origins = sorted({m.origin for m in moves})
            print("Origens válidas:", origins)
            origin = int(input("Origem: "))
            candidates = [m for m in moves if m.origin == origin]
            if not candidates:
                print("Jogada inválida")
                continue
            move = candidates[0]
            if len(candidates) > 1:
                print("Há opções de captura adicionais; CLI usa a primeira. Use a GUI para escolher visualmente.")
            game.apply_move(move)
        else:
            move = agent.select_move(game)
            print("Agente:", move)
            game.apply_move(move)
    print_board(game)
    print("Vencedor:", game.winner)


if __name__ == "__main__":
    main()
