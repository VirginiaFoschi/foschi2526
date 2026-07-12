"""
Loads the Ember Energy day-ahead price CSV , cleans it onto a strict hourly
grid, and exposes chronological train / validation splits plus the set of
valid "episode anchor days" (calendar days for which a full arrival -> 07:00
departure window of hourly prices exists).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EpisodeWindow:
    """Everything needed to run one overnight charging episode."""
    date: pd.Timestamp          # calendar day of arrival
    arrival_hour: int           # 17..20
    prices: np.ndarray          # EUR/MWh, one value per hour, arrival -> 06:00 (inclusive)
    hours: np.ndarray           # hour-of-day (0..23) aligned with `prices`
    n_hours: int                # len(prices)


class PriceCSVLoader:
    """
    Loads a CSV with columns 'Datetime (Local)' and 'Price (EUR/MWhe)', and splits it 
    chronologically into train / validation periods.
    """

    def __init__(
        self,
        csv_path: str,
        *,
        train_years: Tuple[int, int] = (2022, 2024),
        val_years: Tuple[int, int] = (2025, 2025),
        arrival_hour_range: Tuple[int, int] = (17, 20),  # inclusive
        departure_hour: int = 7,
    ):
        self.csv_path = csv_path
        self.train_years = train_years
        self.val_years = val_years
        self.arrival_hour_range = arrival_hour_range
        self.departure_hour = departure_hour

        self.df = self._load_and_clean(csv_path)

        self.train_df = self._slice_years(self.df, train_years)
        self.val_df = self._slice_years(self.df, val_years)

        self.episodes: Dict[str, List[EpisodeWindow]] = {
            "train": self._build_episode_windows(self.train_df),
            "val": self._build_episode_windows(self.val_df),
        }

    @staticmethod
    def _load_and_clean(csv_path: str) -> pd.DataFrame:
        raw = pd.read_csv(csv_path)
        raw["dt"] = pd.to_datetime(raw["Datetime (Local)"])
        df = raw[["dt", "Price (EUR/MWhe)"]].rename(
            columns={"Price (EUR/MWhe)": "price"}
        )
        df = df.sort_values("dt").drop_duplicates(subset="dt").set_index("dt")

        full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h")
        df = df.reindex(full_idx)
        df["price"] = df["price"].interpolate(limit=3)  # fixes rare DST gaps
        df.index.name = "dt"
        return df

    @staticmethod
    def _slice_years(df: pd.DataFrame, years: Tuple[int, int]) -> pd.DataFrame:
        lo, hi = years
        return df[(df.index.year >= lo) & (df.index.year <= hi)].copy()

    def _build_episode_windows(self, price_df: pd.DataFrame) -> List[EpisodeWindow]:
        """Every calendar day in `price_df` for which arrival(17-20h) -> 07:00
        next day is fully covered by hourly prices becomes a candidate episode.
        The actual arrival hour is sampled at reset() time, not here — this
        just pre-slices, per day, the widest possible window (17:00 -> 07:00)
        so that any arrival_hour in the allowed range can be selected later
        without re-touching the source dataframe."""
        windows = []
        if len(price_df) == 0:
            return windows

        days = pd.date_range(
            price_df.index.min().normalize(), price_df.index.max().normalize(), freq="D"
        )
        lo_h, hi_h = self.arrival_hour_range
        for day in days:
            earliest_arrival = day + pd.Timedelta(hours=lo_h)
            departure = day + pd.Timedelta(days=1, hours=self.departure_hour)
            full_window = price_df.loc[earliest_arrival: departure - pd.Timedelta(hours=1)]
            expected_len = int((departure - earliest_arrival).total_seconds() // 3600)
            if len(full_window) != expected_len or full_window["price"].isna().any():
                continue
            windows.append(
                EpisodeWindow(
                    date=day,
                    arrival_hour=lo_h,  # placeholder; sliced further per-episode in the env
                    prices=full_window["price"].to_numpy(dtype=np.float32),
                    hours=full_window.index.hour.to_numpy(),
                    n_hours=len(full_window),
                )
            )
        return windows

    def sample_episode(self, split: str, rng: np.random.Generator) -> EpisodeWindow:
        """Sample a random day from the split, then a random arrival hour
        within `arrival_hour_range`, and slice out the actual price/hour
        arrays the episode will use. Used for TRAINING resets, where visiting
        nights in random order (with repetition) is desirable."""
        candidates = self.episodes[split]
        if not candidates:
            raise RuntimeError(f"No valid episodes available for split '{split}'.")

        base = candidates[int(rng.integers(0, len(candidates)))]
        return self._slice_arrival(base, rng)

    def chronological_episodes(self, split: str, rng: np.random.Generator) -> List[EpisodeWindow]:
        """Every candidate night in the split, IN DATE ORDER, exactly once —
        only the arrival hour is randomized (via `rng`, for reproducibility).
        Use this for evaluating/plotting a policy's performance over time;
        `sample_episode` alone would give a random walk that can repeat or
        skip nights, which is fine for training but meaningless for a
        chronological cumulative-reward plot."""
        candidates = self.episodes[split]
        return [self._slice_arrival(base, rng) for base in candidates]

    def _slice_arrival(self, base: EpisodeWindow, rng: np.random.Generator) -> EpisodeWindow:
        lo_h, hi_h = self.arrival_hour_range
        arrival_hour = int(rng.integers(lo_h, hi_h + 1))
        offset = arrival_hour - lo_h  # how many hours into the pre-sliced window we start
        prices = base.prices[offset:]
        hours = base.hours[offset:]
        return EpisodeWindow(
            date=base.date,
            arrival_hour=arrival_hour,
            prices=prices,
            hours=hours,
            n_hours=len(prices),
        )

    def overnight_train_prices(self) -> np.ndarray:
        """All training-set prices within the 17:00-07:00 window — used to fit
        price quantizers / thresholds without ever touching validation data."""
        mask = (self.train_df.index.hour >= 17) | (self.train_df.index.hour < self.departure_hour)
        return self.train_df.loc[mask, "price"].to_numpy(dtype=np.float32)
