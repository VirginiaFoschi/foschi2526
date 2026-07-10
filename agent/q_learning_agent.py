from __future__ import annotations

import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from .utils import Experience, QEpsGreedyAgent, Schedule


class QLearning(QEpsGreedyAgent):

    def __init__(
            self,
            obs_space_dims: int,
            action_space_dims: int,
            update_coefficient: Union[float, Schedule],
            discount: float = 0.9,
            eps: Union[float, Schedule] = 0.01,
            qval_init: float = 0.,
            seed: Optional[int] = None
    ):
        assert 0 < action_space_dims
        assert isinstance(action_space_dims, int)

        if not isinstance(update_coefficient, Schedule):
            assert 0. < update_coefficient <= 1.

        if not isinstance(eps, Schedule):
            assert 0 <= eps <= 1

        super().__init__(
            obs_space_dims=obs_space_dims,
            action_space_dims=action_space_dims,
            discount=discount,
            eps=eps,
            seed=seed
        )

        self.t = 0
        self.update_coefficient = update_coefficient
        self.qval_init = qval_init

    def initialize(self):
        if isinstance(self.eps, Schedule):
            # Reset noise to starting exploration
            self.eps.initialize()

        # Initialize Q[si][aj] = qval_init
        self.Q = defaultdict(lambda: [self.qval_init] * self.action_space_dims)
        self.Q_update_count = defaultdict(lambda: [0] * self.action_space_dims)
        self.t = 0

    def reset(self):
        # The agent here is prepared for a new episode
        self.t = 0

    def step(self, experience: Experience, **kwargs) -> float:
        s, a, r, sp, done = (
            experience.s, experience.a,
            experience.r, experience.sp,
            experience.done
        )

        if isinstance(s, np.ndarray):
            s = tuple(s)

        if isinstance(sp, np.ndarray):
            sp = tuple(sp)

        if isinstance(self.update_coefficient, Schedule):
            self.update_coefficient.step()
            alpha = self.update_coefficient.value
        else:
            alpha = self.update_coefficient

        if isinstance(self.discount, Schedule):
            self.discount.step()
            gamma = self.discount.value
        else:
            gamma = self.discount

        # Directly estimate q*
        tgt = r + gamma * max(self.Q[sp]) * (1 - done)
        td_error = tgt - self.Q[s][a]

        self.Q[s][a] += alpha * td_error
        self.Q_update_count[s][a] += 1

        if isinstance(self.eps, Schedule):
            self.eps.step()

        self.t += 1

        return td_error

    def get_parameter_to_save(self) -> Dict:
        if isinstance(self.discount, Schedule):
            discount = self.discount.value
        else:
            discount = self.discount

        if isinstance(self.eps, Schedule):
            eps = self.eps.value
        else:
            eps = self.eps

        payload = dict(
            action_space_dims=int(self.action_space_dims),
            Q=dict(self.Q),
            Q_update_count=dict(self.Q_update_count),
            discount=float(discount),
            eps=float(eps),
            step=self.t,
        )

        return payload

    def set_parameter_from_save(self, params: Dict):

        self.action_space_dims = params["action_space_dims"]
        self.Q = defaultdict(lambda: np.zeros(self.action_space_dims))
        self.Q.update(params["Q"])

        self.Q_update_count = defaultdict(
            lambda: np.zeros(self.action_space_dims))
        self.Q_update_count.update(params["Q_update_count"])

        #   This does not recover the schedule if it used one
        #   For now just treating it as a float.
        self.discount = float(params["discount"])
        self.eps = float(params["eps"])
        self.t = params["step"]

    @classmethod
    def save(
            cls,
            agent_instance: "QLearning",
            path,
            envconstructkwargs: Dict = None,  # Env construction parameters
            filename: str = None
    ):
        assert isinstance(agent_instance, QLearning)

        obs_space_dims = agent_instance.obs_space_dims
        action_space_dims = agent_instance.action_space_dims
        qval_init = agent_instance.qval_init
        seed = agent_instance.seed

        if isinstance(agent_instance.update_coefficient, Schedule):
            update_coefficient = agent_instance.update_coefficient.value
        else:
            update_coefficient = agent_instance.update_coefficient

        if isinstance(agent_instance.discount, Schedule):
            discount = agent_instance.discount.value
        else:
            discount = agent_instance.discount

        if isinstance(agent_instance.eps, Schedule):
            eps = agent_instance.eps.value
        else:
            eps = agent_instance.eps

        params = dict(
            obs_space_dims=obs_space_dims,
            action_space_dims=int(action_space_dims),
            qval_init=qval_init,
            seed=seed,
            update_coefficient=update_coefficient,
            Q=dict(agent_instance.Q),
            Q_update_count=dict(agent_instance.Q_update_count),
            discount=float(discount),
            eps=float(eps),
            step=agent_instance.t,
        )

        payload = dict(
            params=params,
            envconstructkwargs=envconstructkwargs,
        )

        file_path = Path(path)
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        # Save instance kwargs
        filename = filename if filename is not None else 'agent_params.pkl'
        instance_file_name = os.path.join(file_path, filename)
        with open(instance_file_name, "wb") as f:
            pickle.dump(payload, f)

        print(
            'Saved Q-Learning instance in: {}/{}'.format(
                file_path, filename
            )
        )

    @classmethod
    def load(cls, path, filename: str = None) -> Tuple["QLearning", Dict]:
        file_path = Path(path)
        filename = filename if filename is not None else 'agent_params.pkl'
        with open(os.path.join(file_path, filename), "rb") as f:
            payload = pickle.load(f)

        params = payload["params"]
        envconstructkwargs = payload["envconstructkwargs"]

        agent_instance = QLearning(
            obs_space_dims=params["obs_space_dims"],
            action_space_dims=params["action_space_dims"],
            qval_init=params["qval_init"],
            seed=params["seed"],
            update_coefficient=params["update_coefficient"]
        )

        agent_instance.Q = defaultdict(
            lambda: np.zeros(agent_instance.action_space_dims))
        agent_instance.Q.update(params["Q"])

        agent_instance.Q_update_count = defaultdict(
            lambda: np.zeros(agent_instance.action_space_dims)
        )
        agent_instance.Q_update_count.update(
            params["Q_update_count"])

        #   This does not recover the schedule if it used one
        #   For now just treating it as a float.
        agent_instance.discount = float(params["discount"])
        agent_instance.eps = float(params["eps"])
        agent_instance.t = params["step"]

        return agent_instance, envconstructkwargs
