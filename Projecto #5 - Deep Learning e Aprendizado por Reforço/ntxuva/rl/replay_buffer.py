from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_mask: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000, seed: int = 42):
        self.data: deque[Transition] = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(self, transition: Transition) -> None:
        self.data.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return self.rng.sample(self.data, batch_size)

    def __len__(self) -> int:
        return len(self.data)
