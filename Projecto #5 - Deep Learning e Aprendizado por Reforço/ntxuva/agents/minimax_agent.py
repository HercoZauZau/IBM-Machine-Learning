from __future__ import annotations

import math

from ntxuva.game.game import NtxuvaGame
from ntxuva.game.move import Move


class MinimaxAgent:

    name = "MiniMax"

    def __init__(self, depth: int = 2, max_children: int = 36):
        self.depth = depth
        self.max_children = max_children
        self.nodes = 0

    def select_move(self, game: NtxuvaGame) -> Move:
        moves = self._ordered_moves(game, game.legal_moves())[: self.max_children]
        if not moves:
            raise RuntimeError("Sem jogadas válidas")
        self.nodes = 0
        best_move = moves[0]
        best_value = -math.inf
        alpha, beta = -math.inf, math.inf
        root_player = game.current_player

        for move in moves:
            child = game.clone()
            child.apply_move(move)
            value = -self._negamax(child, self.depth - 1, -beta, -alpha, 1 - root_player)
            if value > best_value:
                best_value = value
                best_move = move
            alpha = max(alpha, value)
        return best_move

    def _negamax(self, game: NtxuvaGame, depth: int, alpha: float, beta: float, perspective: int) -> float:
        self.nodes += 1
        if depth <= 0 or game.terminal:
            return self.evaluate(game, perspective)

        moves = self._ordered_moves(game, game.legal_moves())[: self.max_children]
        if not moves:
            return self.evaluate(game, perspective)

        value = -math.inf
        for move in moves:
            child = game.clone()
            child.apply_move(move)
            score = -self._negamax(child, depth - 1, -beta, -alpha, 1 - perspective)
            value = max(value, score)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value

    def evaluate(self, game: NtxuvaGame, player: int) -> float:
        opp = 1 - player
        if game.terminal:
            if game.winner == player:
                return 10000.0
            if game.winner == opp:
                return -10000.0
            return 0.0

        capture_diff = float(game.captured[player] - game.captured[opp])
        own_seeds = sum(int(game.board[r, c]) for r, c in game.sow_path(player))
        opp_seeds = sum(int(game.board[r, c]) for r, c in game.sow_path(opp))

        # Mobilidade calculada em clones para evitar alterar a vez.
        own_mobility = len(game.legal_moves(player))
        opp_mobility = len(game.legal_moves(opp))

        own_inner, _ = game.rows_for_player(player)
        opp_inner, _ = game.rows_for_player(opp)
        inner_diff = int(game.board[own_inner].sum()) - int(game.board[opp_inner].sum())

        return (
            20.0 * capture_diff
            + 0.5 * (own_seeds - opp_seeds)
            + 0.8 * (own_mobility - opp_mobility)
            + 0.2 * inner_diff
        )

    def _ordered_moves(self, game: NtxuvaGame, moves: list[Move]) -> list[Move]:
        """Ordena capturas sem clonar/aplicar cada jogada.

        A semeadura depende apenas da origem; por isso é calculada uma vez por
        origem e o ganho das opções de captura é estimado directamente.
        """
        player = game.current_player
        opp = 1 - player
        by_origin: dict[int, tuple[object, list[tuple[int, int]]]] = {}
        scored: list[tuple[int, Move]] = []
        for move in moves:
            if move.origin not in by_origin:
                board, landing, _relay, _cycle = game._simulate_sowing(player, move.origin)
                base, _available = game._capture_base_and_available_extras(player, board, landing)
                by_origin[move.origin] = (board, base)
            board, base = by_origin[move.origin]
            gain = sum(int(board[r, c]) for r, c in base)
            gain += sum(int(board[r, c]) for r, c in (game.local_to_global(opp, i) for i in move.extra_captures))
            scored.append((gain, move))
        scored.sort(key=lambda x: (x[0], -x[1].origin), reverse=True)
        return [m for _, m in scored]
