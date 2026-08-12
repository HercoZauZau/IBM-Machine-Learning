from __future__ import annotations

import random
from collections import defaultdict

import numpy as np

from ntxuva.game.game import NtxuvaGame
from ntxuva.game.move import Move


class TabularQAgent:
    """Baseline didáctico de Q-Learning.

    Não é recomendado como agente final devido ao enorme espaço de estados.
    Usa uma representação inteira do estado e uma tabela esparsa.
    """

    name = "Q-Learning (tabular)"

    def __init__(self, alpha: float = 0.1, gamma: float = 0.99, epsilon: float = 0.1, seed: int = 42):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.q: dict[tuple[int, ...], dict[int, float]] = defaultdict(dict)

    @staticmethod
    def state_key(game: NtxuvaGame) -> tuple[int, ...]:
        p = game.current_player
        opp = 1 - p
        own = [int(game.board[r, c]) for r, c in game.sow_path(p)]
        other = [int(game.board[r, c]) for r, c in game.sow_path(opp)]
        return tuple(own + other + [int(game.captured[p]), int(game.captured[opp])])

    def select_action(self, game: NtxuvaGame, epsilon: float | None = None) -> int:
        eps = self.epsilon if epsilon is None else epsilon
        legal = game.legal_action_ids()
        if self.rng.random() < eps:
            return self.rng.choice(legal)
        key = self.state_key(game)
        values = self.q[key]
        return max(legal, key=lambda a: values.get(a, 0.0))

    def select_move(self, game: NtxuvaGame, epsilon: float | None = None) -> Move:
        return game.codec.decode(self.select_action(game, epsilon))
