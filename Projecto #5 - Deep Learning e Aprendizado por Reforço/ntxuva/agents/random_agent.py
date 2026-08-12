from __future__ import annotations

import random

from ntxuva.game.game import NtxuvaGame
from ntxuva.game.move import Move


class RandomAgent:
    name = "Random"

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def select_move(self, game: NtxuvaGame) -> Move:
        moves = game.legal_moves()
        if not moves:
            raise RuntimeError("Sem jogadas válidas")
        return self.rng.choice(moves)
