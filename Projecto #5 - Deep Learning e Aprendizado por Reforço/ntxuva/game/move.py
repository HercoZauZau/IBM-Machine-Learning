from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Move:
    """Jogada completa.

    origin é uma casa local do jogador (0..15 num tabuleiro 4x8).
    extra_captures contém 0, 1 ou 2 casas locais do adversário escolhidas
    como capturas adicionais, quando aplicável.
    """

    origin: int
    extra_captures: tuple[int, ...] = ()
