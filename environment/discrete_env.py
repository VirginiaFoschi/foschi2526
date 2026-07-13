from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Any, Optional
from dataclasses import dataclass

from .model_env import EVBatteryModel
from data.price_loader import PriceCSVLoader, EpisodeWindow


class Quantizer:
    """
    converts continous numbers into discrete bins (e.g. for SoC and price) for the tabular Q-learning agent."""

    def __init__(
        self,
        x_ref,
        *,
        num_bins: int,
        mode: str = "equal_width",
        range_pct: float = 100,
        log_scale: bool = False,
    ):
        self.num_bins = int(num_bins)
        self.mode = str(mode)
        self.range_pct = float(range_pct)
        self.log_scale = bool(log_scale)

        x_ref = np.asarray(x_ref)

        if self.log_scale:
            if np.any(x_ref < 0):
                raise ValueError("log_scale=True requires x_ref >= 0")
            x_t = np.log1p(x_ref)
        else:
            x_t = x_ref

        if self.mode == "equal_width":
            if not (0 < self.range_pct <= 100):
                raise ValueError("range_pct must be in (0, 100]")
            lo = np.percentile(x_t, (100 - self.range_pct) / 2)
            hi = np.percentile(x_t, 100 - (100 - self.range_pct) / 2)
            self.edges = np.linspace(lo, hi, self.num_bins + 1)

        elif self.mode == "percentile":
            self.edges = np.percentile(x_t, np.linspace(0, 100, self.num_bins + 1))

        else:
            raise ValueError("mode must be 'equal_width' or 'percentile'")

        self.centers_t = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.centers = np.expm1(self.centers_t) if self.log_scale else self.centers_t

    def __call__(self, x):
        x = np.asarray(x)
        x_t = np.log1p(x) if self.log_scale else x
        idx = np.searchsorted(self.edges, x_t, side="right") - 1
        idx = np.clip(idx, 0, len(self.centers_t) - 1)
        if self.log_scale:
            return np.expm1(self.centers_t[idx])
        return self.centers_t[idx]

    def bin_index(self, x):
        """Return 0..num_bins-1 bin index (clipped) for x."""
        x = np.asarray(x)
        x_t = np.log1p(x) if self.log_scale else x
        idx = np.searchsorted(self.edges, x_t, side="right") - 1
        return int(np.clip(idx, 0, self.num_bins - 1))


@dataclass(frozen=True)
class DiscreteState:
    """
    (soc_bucket, hours_until_departure, price_bin)

    SoC and price are stored quantized to integer bin indices
    """
    soc_bin: int
    hours_until_departure: int
    price_bin: int


class EVChargingEnvDiscrete(gym.Env):
    """
    One episode = one overnight charging session (from arrival to departure).
      - Arrival: random hour in [17, 20], random SoC in `arrival_soc_range`.
      - Departure: fixed at 07:00 the next day.
      - Action (per hour): 0 = idle, 1 = charge at 50% (5.5 kW), 2 = charge at 100% (11 kW).
      - Reward (per hour): -cost of energy drawn from the grid this hour (EUR);
        price can be negative, in which case the agent is paid to charge.
        A one-off penalty for any shortfall below `target_soc` is added to the
        reward on the final step of the episode.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        prices_csv_path: str,
        split: str = "train",                       # "train" or "val"
        train_years: Tuple[int, int] = (2022, 2024),
        val_years: Tuple[int, int] = (2025, 2025),
        battery_capacity_kwh: float = 60.0,
        charging_efficiency: float = 0.90,
        target_soc: float = 0.80,
        shortfall_penalty_per_kwh: float = 2.0,
        arrival_hour_range: Tuple[int, int] = (17, 20),
        departure_hour: int = 7,
        arrival_soc_range: Tuple[float, float] = (0.15, 0.45),
        price_n_bins: int = 3,
        soc_n_bins: int = 10,
        price_bin_mode: str = "percentile",           # "percentile" or "equal_width"
        loader: Optional[PriceCSVLoader] = None,       # reuse a loader across train/val envs
        quantizers: Optional[Dict[str, Quantizer]] = None,  # reuse quantizers (fit on train only!)
    ):
        super().__init__()

        assert split in ("train", "val")
        self.split = split

        self.battery_capacity_kwh = battery_capacity_kwh
        self.charging_efficiency = charging_efficiency
        self.target_soc = target_soc
        self.shortfall_penalty_per_kwh = shortfall_penalty_per_kwh
        self.arrival_soc_range = arrival_soc_range

        # price data: reuse a shared loader if one is passed in, so a
        # train env and a val env agree on the exact same split boundaries (also to avoid unecessary work)
        self.loader = loader if loader is not None else PriceCSVLoader(
            prices_csv_path,
            train_years=train_years,
            val_years=val_years,
            arrival_hour_range=arrival_hour_range,
            departure_hour=departure_hour,
        )

        # action space: 0=idle, 1=half, 2=full
        self.action_space = spaces.Discrete(3)
        self.battery_model_cls = EVBatteryModel

        # quantizers: quantizer looks just at training data to create bins
        if quantizers is None: #in the training env it will enter here
            train_overnight_prices = self.loader.overnight_train_prices()
            self.quantizers = {
                "price": Quantizer(
                    train_overnight_prices,
                    num_bins=price_n_bins,
                    mode=price_bin_mode,
                    range_pct=100,
                    log_scale=False,
                ),
                "soc": Quantizer(
                    np.linspace(0.0, 1.0, 1000),
                    num_bins=soc_n_bins,
                    mode="equal_width",
                    range_pct=100,
                    log_scale=False,
                ),
            }
        else: #in the validation env will enter here, reusing the quantizers fitted on the training data
            self.quantizers = quantizers

        # observation space (DiscreteState)
        max_hours = (24 - arrival_hour_range[0]) + departure_hour
        self.observation_space = spaces.MultiDiscrete(
            [soc_n_bins, max_hours + 1, price_n_bins]
        )

        self._episode: Optional[EpisodeWindow] = None
        self._battery: Optional[EVBatteryModel] = None
        self._t = 0  # index into the current episode's price array

    #called at the beginning at the beginningof every episode
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[DiscreteState, Dict[str, Any]]:
        super().reset(seed=seed)

        options = options or {}
        initial_soc = options.get("initial_soc", None)
        episode_window = options.get("episode_window", None)

        self._episode = episode_window if episode_window is not None \
            else self.loader.sample_episode(self.split, self.np_random)

        if initial_soc is None:
            lo, hi = self.arrival_soc_range
            initial_soc = float(self.np_random.uniform(lo, hi)) #sample from arrival soc range

        self._battery = self.battery_model_cls(
            self.battery_capacity_kwh, self.charging_efficiency, initial_soc
        )
        self._t = 0 #current hour = arrival hour

        obs = self._get_obs()
        info = self._get_info(action=None, cost_eur=0.0, meta={"capped": False})
        return obs, info

    #called once every hour until the episode finishes (ie until the car leaves the charging station)
    def step(self, action: int) -> Tuple[DiscreteState, float, bool, bool, Dict[str, Any]]:
        assert self._episode is not None, "Call reset() before step()."
        if isinstance(action, (np.integer,)):
            action = int(action)
        assert action in (0, 1, 2), f"invalid action: {action}"

        price = float(self._episode.prices[self._t])
        energy_kwh, meta = self._battery.step(action, delta_time_h=1.0)
        cost_eur = self._battery.cost_eur(energy_kwh, price)
        reward = -cost_eur

        self._t += 1
        terminated = False  # no failure state — shortfall is penalized, not terminal
        truncated = self._t >= self._episode.n_hours

        if truncated:
            shortfall_kwh = self._battery.shortfall_kwh(self.target_soc)
            penalty = shortfall_kwh * self.shortfall_penalty_per_kwh
            reward -= penalty
            cost_eur += penalty

        obs = self._get_obs()
        info = self._get_info(action=action, cost_eur=cost_eur, meta=meta)
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> DiscreteState:
        hours_until_departure = self._episode.n_hours - self._t
        # hour-of-day for the NEXT decision point (or last hour if episode just ended)
        idx = min(self._t, self._episode.n_hours - 1)
        price = float(self._episode.prices[idx])

        return DiscreteState(
            soc_bin=self.quantizers["soc"].bin_index(self._battery.soc),
            hours_until_departure=int(hours_until_departure),
            price_bin=self.quantizers["price"].bin_index(price),
        )

    def _get_info(self, *, action, cost_eur, meta) -> Dict[str, Any]:
        return dict(
            step=self._t,
            date=str(self._episode.date.date()),
            arrival_hour=self._episode.arrival_hour,
            action=action,
            soc=self._battery.soc,
            cost_eur=cost_eur,
            capped=meta.get("capped", False),
        )

    def close(self):
        pass


def make_train_val_envs(
    prices_csv_path: str,
    *,
    train_years: Tuple[int, int] = (2021, 2023),
    val_years: Tuple[int, int] = (2024, 2025),
    **env_kwargs: Any,
) -> Tuple[EVChargingEnvDiscrete, EVChargingEnvDiscrete]:
    """
    Convenience constructor that guarantees the train and validation
    environments share:
      - the exact same cleaned price dataframe / split boundaries (one
        PriceCSVLoader instance), and
      - the exact same price/SoC quantizers, fit ONLY on training data.
    """
    train_env = EVChargingEnvDiscrete(
        prices_csv_path=prices_csv_path,
        split="train",
        train_years=train_years,
        val_years=val_years,
        **env_kwargs,
    )
    val_env = EVChargingEnvDiscrete(
        prices_csv_path=prices_csv_path,
        split="val",
        train_years=train_years,
        val_years=val_years,
        loader=train_env.loader,
        quantizers=train_env.quantizers,
        **env_kwargs,
    )
    return train_env, val_env


if __name__ == "__main__":
    # Smoke test: random policy for a handful of episodes on train + val.
    train_env, val_env = make_train_val_envs("./Germany.csv")

    for name, env in [("train", train_env), ("val", val_env)]:
        rng = np.random.default_rng(0)
        obs, info = env.reset(seed=0)
        total_reward = 0.0
        n_steps = 0
        terminated = truncated = False
        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            n_steps += 1
        print(f"[{name}] episode date={info['date']} arrival={info['arrival_hour']}h "
              f"steps={n_steps} final_soc={info['soc']:.3f} total_reward={total_reward:.2f}")
