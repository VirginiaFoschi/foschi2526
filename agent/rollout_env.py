from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from environment.discrete_env import EVChargingEnvDiscrete, make_train_val_envs
from .baseline import BasePolicy, NaiveImmediateChargePolicy


@dataclass
class RolloutSignals:
    """Step-level signals concatenated across many episodes (nights), plus
    per-episode summaries"""

    t: np.ndarray                    # global step index (0..N-1), across all episodes
    episode_idx: np.ndarray          # which episode (0..n_episodes-1) each step belongs to

    action: np.ndarray
    price: np.ndarray
    soc: np.ndarray                  # raw (continuous) SoC, from info — for plotting only
    hours_until_departure: np.ndarray
    hour_of_day: np.ndarray
    reward: np.ndarray
    cumulative_reward: np.ndarray
    capped: np.ndarray

    episode_boundaries: np.ndarray   # global step index where each new episode starts
    episode_total_reward: np.ndarray  # one value per episode
    episode_cumulative_reward: np.ndarray  # cumsum of episode_total_reward


def rollout(
    *,
    env: EVChargingEnvDiscrete,
    policy: BasePolicy,
    episode_windows: Optional[list] = None,
    seed: Optional[int] = None,
) -> RolloutSignals:
    """
    Roll out overnight charging sessions back-to-back, recording every step.
    Each episode is an independent env.reset() (a different historical night,
    arrival hour, and arrival SoC), concatenated here purely for
    plotting/analysis convenience
    """
    if (episode_windows is None):
        raise ValueError("Pass exactly one of `n_episodes` or `episode_windows`.")

    n_episodes = len(episode_windows)

    all_action, all_price, all_soc = [], [], []
    all_hour_of_day, all_hleft, all_reward, all_capped = [], [], [], []
    episode_idx: List[int] = []
    episode_boundaries = []
    episode_total_reward = []

    global_t = 0
    for ep in range(n_episodes):
        ep_seed = None if seed is None else seed + ep
        reset_options = {"episode_window": episode_windows[ep]} if episode_windows is not None else None
        obs, info = env.reset(seed=ep_seed, options=reset_options)
        episode_boundaries.append(global_t)

        terminated = truncated = False
        ep_reward = 0.0
        while not (terminated or truncated):
            action = policy.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            all_action.append(action)
            all_price.append(env._episode.prices[info["step"] - 1])  # price of the hour just consumed
            all_soc.append(info["soc"])
            all_hleft.append(obs.hours_until_departure)
            all_hour_of_day.append(info["hour_of_day"])
            all_reward.append(reward)
            all_capped.append(info["capped"])
            episode_idx.append(ep)

            ep_reward += reward
            obs = next_obs
            global_t += 1

        episode_total_reward.append(ep_reward)

    reward_arr = np.asarray(all_reward, dtype=float)
    episode_total_reward = np.asarray(episode_total_reward, dtype=float)
    return RolloutSignals(
        t=np.arange(global_t, dtype=int),
        episode_idx=np.asarray(episode_idx, dtype=int),
        action=np.asarray(all_action, dtype=int),
        price=np.asarray(all_price, dtype=float),
        soc=np.asarray(all_soc, dtype=float),
        hours_until_departure=np.asarray(all_hleft, dtype=int),
        hour_of_day=np.asarray(all_hour_of_day, dtype=int),
        reward=reward_arr,
        cumulative_reward=np.cumsum(reward_arr),
        capped=np.asarray(all_capped, dtype=bool),
        episode_boundaries=np.asarray(episode_boundaries, dtype=int),
        episode_total_reward=episode_total_reward,
        episode_cumulative_reward=np.cumsum(episode_total_reward),
    )
