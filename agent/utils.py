
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, NamedTuple, Optional

import numpy as np


class Experience(NamedTuple):
    s: Any     # state (must be hashable, e.g. a tuple)
    a: int     # action taken
    r: float   # reward received
    sp: Any    # next state (hashable)
    done: bool  # True if sp is terminal / episode ended here


class Schedule(ABC):
    """A scalar value that changes each time .step() is called."""

    @property
    @abstractmethod
    def value(self) -> float:
        ...

    @abstractmethod
    def step(self) -> None:
        ...

    @abstractmethod
    def initialize(self) -> None:
        """Reset the schedule back to its starting value."""
        ...


class LinearSchedule(Schedule):
    """Linearly interpolates from `start` to `end` over `n_steps` calls to
    .step(), then holds at `end`."""

    def __init__(self, start: float, end: float, n_steps: int):
        assert n_steps > 0
        self.start = start
        self.end = end
        self.n_steps = n_steps
        self._t = 0
        self._value = start

    def initialize(self) -> None:
        self._t = 0
        self._value = self.start

    def step(self) -> None:
        self._t += 1
        frac = min(1.0, self._t / self.n_steps)
        self._value = self.start + frac * (self.end - self.start)

    @property
    def value(self) -> float:
        return self._value


class QEpsGreedyAgent(ABC):
    """Abstract tabular agent: holds self.Q (defaultdict: state -> list of
    action-values) and selects actions epsilon-greedily wrt it"""

    def __init__(
        self,
        obs_space_dims: int,
        action_space_dims: int,
        discount: float = 0.9,
        eps=0.01,
        seed: Optional[int] = None,
    ):
        self.obs_space_dims = obs_space_dims
        self.action_space_dims = action_space_dims
        self.discount = discount
        self.eps = eps
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.Q = None  # set by initialize()
        self.initialize()

    @abstractmethod
    def initialize(self):
        ...

    @abstractmethod
    def reset(self):
        ...

    @abstractmethod
    def step(self, experience: Experience, **kwargs) -> float:
        ...

    def act(self, obs, greedy: bool = False) -> int:
        """Epsilon-greedy action selection. `obs` must be hashable (e.g. a
        tuple). Pass greedy=True at evaluation time to disable exploration."""
        if isinstance(obs, np.ndarray):
            obs = tuple(obs)

        eps_value = self.eps.value if isinstance(self.eps, Schedule) else self.eps

        if (not greedy) and self.rng.random() < eps_value:
            return int(self.rng.integers(0, self.action_space_dims))

        q_values = self.Q[obs]
        return int(np.argmax(q_values))
