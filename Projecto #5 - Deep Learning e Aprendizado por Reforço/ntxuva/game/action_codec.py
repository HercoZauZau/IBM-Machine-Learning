from __future__ import annotations

from itertools import combinations

from .move import Move


class ActionCodec:
    """Mapeia jogadas compostas para um espaço discreto fixo.

    Para cada origem existem opções de captura adicional:
    - nenhuma;
    - uma das 16 casas do adversário;
    - qualquer par de casas do adversário.

    Em 4x8: 16 * (1 + 16 + C(16,2)) = 2192 acções.
    """

    def __init__(self, pits_per_player: int = 16):
        self.pits_per_player = pits_per_player
        capture_options: list[tuple[int, ...]] = [()]
        capture_options += [(i,) for i in range(pits_per_player)]
        capture_options += list(combinations(range(pits_per_player), 2))
        self.capture_options = tuple(capture_options)
        self._capture_to_slot = {c: i for i, c in enumerate(self.capture_options)}
        self.options_per_origin = len(self.capture_options)
        self.action_size = pits_per_player * self.options_per_origin

    def encode(self, move: Move) -> int:
        extras = tuple(sorted(move.extra_captures))
        slot = self._capture_to_slot[extras]
        return move.origin * self.options_per_origin + slot

    def decode(self, action: int) -> Move:
        if action < 0 or action >= self.action_size:
            raise ValueError(f"Acção fora do intervalo: {action}")
        origin, slot = divmod(action, self.options_per_origin)
        return Move(origin=origin, extra_captures=self.capture_options[slot])
