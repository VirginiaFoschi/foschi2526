from __future__ import annotations
import numpy as np

from abc import ABC, abstractmethod

from environment.discrete_env import DiscreteState, Quantizer, make_train_val_envs

class BasePolicy(ABC):
    @abstractmethod
    def act(self, obs: DiscreteState) -> int:
        """Return a discrete action: 0=idle, 1=half, 2=full."""
        raise NotImplementedError


class RandomPolicy(BasePolicy):
    def __init__(self, action_space, seed: int = 0):
        self.action_space = action_space
        self.action_space.seed(seed)

    def act(self, obs: DiscreteState) -> int:
        return int(self.action_space.sample())


class NaiveImmediateChargePolicy(BasePolicy):

    def __init__(self, soc_quantizer: Quantizer, target_soc: float):
        self.target_soc_bin = soc_quantizer.bin_index(target_soc)

    def act(self, obs: DiscreteState) -> int:
        return 2 if obs.soc_bin < self.target_soc_bin else 0


class TrainedQPolicy(BasePolicy):

    def __init__(self, agent):
        self.agent = agent

    def act(self, obs: DiscreteState) -> int:
        return self.agent.act(obs, greedy=True)
