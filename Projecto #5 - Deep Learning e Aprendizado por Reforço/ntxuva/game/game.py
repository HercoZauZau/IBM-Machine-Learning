from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np

from .action_codec import ActionCodec
from .move import Move


@dataclass
class MoveResult:
    player: int
    move: Move
    captured: int
    relay_steps: int
    terminal: bool
    winner: int | None
    draw: bool = False


class NtxuvaGame:
    """Motor da variante computacional moçambicana de Ntxuva.

    Convenções principais:
    - tabuleiro 4 x 8 (configurável em colunas);
    - duas sementes por casa no início;
    - cada jogador controla duas filas;
    - semeadura anti-horária apenas nas próprias filas;
    - relay sowing: se a última semente cai numa casa já ocupada,
      recolhem-se as sementes dessa casa e a semeadura continua;
    - o movimento termina quando a última semente cai numa casa vazia;
    - captura ocorre quando isso acontece na fila interna e a casa interna
      oposta do adversário contém sementes;
    - captura as duas casas opostas da coluna e, quando disponíveis, duas
      casas adicionais do adversário escolhidas pelo jogador;
    - quando existe pelo menos uma casa com >=2 sementes, só essas casas
      podem iniciar a jogada; singletons são usados apenas quando não há gula;
    - o jogo termina quando o jogador da vez não tem jogada válida.
    """

    def __init__(self, columns: int = 8, seeds_per_pit: int = 2, max_relay_steps: int = 1000):
        if columns < 2:
            raise ValueError("O tabuleiro deve ter pelo menos 2 colunas.")
        self.columns = columns
        self.seeds_per_pit = seeds_per_pit
        self.max_relay_steps = max_relay_steps
        self.pits_per_player = 2 * columns
        self.codec = ActionCodec(self.pits_per_player)
        self.reset()

    def reset(self) -> np.ndarray:
        self.board = np.full((4, self.columns), self.seeds_per_pit, dtype=np.int16)
        self.captured = np.zeros(2, dtype=np.int16)
        self.current_player = 0
        self.move_count = 0
        self.terminal = False
        self.winner: int | None = None
        self.draw = False
        return self.board.copy()

    def clone(self) -> "NtxuvaGame":
        other = NtxuvaGame(self.columns, self.seeds_per_pit, self.max_relay_steps)
        other.board = self.board.copy()
        other.captured = self.captured.copy()
        other.current_player = self.current_player
        other.move_count = self.move_count
        other.terminal = self.terminal
        other.winner = self.winner
        other.draw = self.draw
        return other

    # ------------------------------------------------------------------
    # Geometria do tabuleiro
    # ------------------------------------------------------------------
    def rows_for_player(self, player: int) -> tuple[int, int]:
        """Retorna (inner_row, outer_row)."""
        return (2, 3) if player == 0 else (1, 0)

    def sow_path(self, player: int) -> list[tuple[int, int]]:
        """Ciclo anti-horário visto da perspectiva do jogador.

        O percurso é espelhado entre jogadores para manter a mesma orientação
        relativa à posição de cada um no tabuleiro.
        """
        inner, outer = self.rows_for_player(player)
        if player == 0:
            return [(outer, c) for c in range(self.columns)] + [
                (inner, c) for c in range(self.columns - 1, -1, -1)
            ]
        return [(outer, c) for c in range(self.columns - 1, -1, -1)] + [
            (inner, c) for c in range(self.columns)
        ]

    def local_to_global(self, player: int, local_index: int) -> tuple[int, int]:
        path = self.sow_path(player)
        if not 0 <= local_index < len(path):
            raise ValueError("Índice local inválido")
        return path[local_index]

    def global_to_local(self, player: int, row: int, col: int) -> int:
        try:
            return self.sow_path(player).index((row, col))
        except ValueError as exc:
            raise ValueError("Casa não pertence ao jogador") from exc

    def opponent_local_for_global(self, player: int, row: int, col: int) -> int:
        return self.global_to_local(1 - player, row, col)

    # ------------------------------------------------------------------
    # Regras e geração de jogadas
    # ------------------------------------------------------------------
    def _candidate_origins(self, player: int) -> list[int]:
        path = self.sow_path(player)
        counts = [int(self.board[r, c]) for r, c in path]
        gula = [i for i, n in enumerate(counts) if n >= 2]
        if gula:
            return gula

        # Regra tchonga: singleton apenas se a próxima casa está vazia.
        singles: list[int] = []
        for i, n in enumerate(counts):
            if n == 1:
                nr, nc = path[(i + 1) % len(path)]
                if self.board[nr, nc] == 0:
                    singles.append(i)
        return singles

    def _simulate_sowing(self, player: int, origin: int) -> tuple[np.ndarray, tuple[int, int], int, bool]:
        """Executa apenas a semeadura num tabuleiro copiado.

        Retorna board, última casa, relay_steps e cycle_detected.
        """
        board = self.board.copy()
        path = self.sow_path(player)
        start = origin
        r, c = path[start]
        hand = int(board[r, c])
        if hand <= 0:
            raise ValueError("Não é possível jogar a partir de uma casa vazia")
        board[r, c] = 0
        idx = start
        relay = 0

        # A configuração completa do subestado ajuda a detectar ciclos reais.
        seen: set[tuple[bytes, int, int]] = set()

        while True:
            while hand > 0:
                idx = (idx + 1) % len(path)
                rr, cc = path[idx]
                board[rr, cc] += 1
                hand -= 1

            rr, cc = path[idx]
            # count == 1 significa que a casa estava vazia antes da última semente.
            if board[rr, cc] == 1:
                return board, (rr, cc), relay, False

            hand = int(board[rr, cc])
            board[rr, cc] = 0
            relay += 1
            key = (board.tobytes(), idx, hand)
            if key in seen or relay >= self.max_relay_steps:
                return board, (rr, cc), relay, True
            seen.add(key)

    def _capture_base_and_available_extras(
        self, player: int, board_after_sow: np.ndarray, landing: tuple[int, int]
    ) -> tuple[list[tuple[int, int]], list[int]]:
        inner, _outer = self.rows_for_player(player)
        row, col = landing
        if row != inner:
            return [], []

        opp = 1 - player
        opp_inner, opp_outer = self.rows_for_player(opp)
        # Captura só é activada se a casa interna directamente oposta tiver sementes.
        if board_after_sow[opp_inner, col] <= 0:
            return [], []

        base: list[tuple[int, int]] = [(opp_inner, col)]
        if board_after_sow[opp_outer, col] > 0:
            base.append((opp_outer, col))

        blocked = set(base)
        extras: list[int] = []
        for local, (r, c) in enumerate(self.sow_path(opp)):
            if (r, c) not in blocked and board_after_sow[r, c] > 0:
                extras.append(local)
        return base, extras

    @staticmethod
    def _extra_options(available: list[int]) -> list[tuple[int, ...]]:
        if len(available) >= 2:
            return [tuple(x) for x in combinations(available, 2)]
        if len(available) == 1:
            return [(available[0],)]
        return [()]

    def legal_moves(self, player: int | None = None) -> list[Move]:
        if self.terminal:
            return []
        player = self.current_player if player is None else player
        moves: list[Move] = []
        for origin in self._candidate_origins(player):
            board, landing, _relay, cycle = self._simulate_sowing(player, origin)
            if cycle:
                # Movimentos detectados como cíclicos são excluídos da versão computacional.
                continue
            _base, available = self._capture_base_and_available_extras(player, board, landing)
            if _base:
                for extras in self._extra_options(available):
                    moves.append(Move(origin, extras))
            else:
                moves.append(Move(origin, ()))
        return sorted(set(moves))

    def legal_action_ids(self, player: int | None = None) -> list[int]:
        return [self.codec.encode(m) for m in self.legal_moves(player)]

    def action_mask(self, player: int | None = None) -> np.ndarray:
        mask = np.zeros(self.codec.action_size, dtype=np.bool_)
        ids = self.legal_action_ids(player)
        mask[ids] = True
        return mask

    # ------------------------------------------------------------------
    # Execução
    # ------------------------------------------------------------------
    def apply_move(self, move: Move) -> MoveResult:
        if self.terminal:
            raise RuntimeError("A partida já terminou")
        legal = set(self.legal_moves())
        if move not in legal:
            raise ValueError(f"Jogada inválida: {move}")

        player = self.current_player
        board, landing, relay, cycle = self._simulate_sowing(player, move.origin)
        if cycle:
            raise ValueError("Jogada cíclica não permitida pela versão computacional")

        base, available = self._capture_base_and_available_extras(player, board, landing)
        captured_now = 0
        if base:
            allowed = set(available)
            if any(i not in allowed for i in move.extra_captures):
                raise ValueError("Captura adicional inválida")
            targets = list(base)
            opp = 1 - player
            targets.extend(self.local_to_global(opp, i) for i in move.extra_captures)
            for r, c in targets:
                captured_now += int(board[r, c])
                board[r, c] = 0

        self.board = board
        self.captured[player] += captured_now
        self.move_count += 1
        self.current_player = 1 - player

        self._update_terminal_status()
        return MoveResult(
            player=player,
            move=move,
            captured=captured_now,
            relay_steps=relay,
            terminal=self.terminal,
            winner=self.winner,
            draw=self.draw,
        )

    def apply_action(self, action: int) -> MoveResult:
        return self.apply_move(self.codec.decode(action))

    def _update_terminal_status(self) -> None:
        if self.legal_moves(self.current_player):
            return
        self.terminal = True
        if self.captured[0] > self.captured[1]:
            self.winner = 0
        elif self.captured[1] > self.captured[0]:
            self.winner = 1
        else:
            self.winner = None
            self.draw = True

    # ------------------------------------------------------------------
    # Estado para IA
    # ------------------------------------------------------------------
    def encode_state(self, perspective: int | None = None) -> np.ndarray:
        """Estado normalizado da perspectiva de um jogador.

        [16 casas próprias, 16 casas adversárias, capturas próprias, capturas adversárias]
        """
        perspective = self.current_player if perspective is None else perspective
        opp = 1 - perspective
        own = [self.board[r, c] for r, c in self.sow_path(perspective)]
        other = [self.board[r, c] for r, c in self.sow_path(opp)]
        scale = max(1.0, float(4 * self.columns * self.seeds_per_pit))
        vec = np.array(
            list(own) + list(other) + [self.captured[perspective], self.captured[opp]],
            dtype=np.float32,
        )
        return vec / scale

    @property
    def state_size(self) -> int:
        return 2 * self.pits_per_player + 2

    def summary(self) -> dict:
        return {
            "current_player": self.current_player,
            "captured": self.captured.tolist(),
            "move_count": self.move_count,
            "terminal": self.terminal,
            "winner": self.winner,
            "draw": self.draw,
            "legal_moves": len(self.legal_moves()) if not self.terminal else 0,
        }
