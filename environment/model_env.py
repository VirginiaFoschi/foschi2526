from typing import Optional, Tuple, Dict, Sequence

import numpy as np
from matplotlib import pyplot as plt


class EVBatteryModel:
    """
    Physical model of a single EV battery for one overnight charging session.
    Actions are discrete power levels (idle / half / full)
    """

    # action index -> charging power (kW)
    POWER_LEVELS_KW = {0: 0.0, 1: 5.5, 2: 11.0}
    ACTION_NAMES = {0: "idle", 1: "half (50%)", 2: "full (100%)"}

    def __init__(
        self,
        battery_capacity_kwh: float = 60.0,
        charging_efficiency: float = 0.90,
        initial_soc: Optional[float] = None,
    ):
        self.battery_capacity_kwh = battery_capacity_kwh
        self.charging_efficiency = charging_efficiency #how much of the electricity from the grid actually ends up stored in the battery

        #how full the battery already is hen charging starts (default to 30% if not specified)
        self.soc = 0.30 if initial_soc is None else float(initial_soc)
        self.initial_soc = self.soc

        assert 0.0 <= self.soc <= 1.0, "initial_soc must be in [0, 1]"

    def step(self, action: int, delta_time_h: float = 1.0) -> Tuple[float, Dict]:
        """
        Action convention:
            0 -> idle       (0 kW)
            1 -> half power (5.5 kW)
            2 -> full power (11 kW)

        :param action: discrete action index (0, 1 or 2)
        :param delta_time_h: duration of this step, in hours
        :return: (cost_eur, meta) — cost_eur is what the agent pays this step
                 (can be negative, i.e. the agent gets paid, when the
                 day-ahead price itself is negative)
        """
        assert action in self.POWER_LEVELS_KW, f"invalid action: {action}"
        assert delta_time_h > 0, "time step must be positive"

        power_kw = self.POWER_LEVELS_KW[action]

        #Energy drawn from the grid in one hour at power P (kW): E [kWh] = P * delta_time_h
        energy_from_grid_kwh = power_kw * delta_time_h #delta_time_h says how much time we charge at that power (in our case is always 1 hour since each action last one hour)

        #Energy actually stored in the battery: E_battery = E_grid * charging_efficiency
        energy_to_battery_kwh = energy_from_grid_kwh * self.charging_efficiency

        new_soc = min(self.soc + energy_to_battery_kwh / self.battery_capacity_kwh, 1.0) #division is used in order to get a value between 0-1 (percentage of the battery capacity)

        meta = dict(
            capped=(new_soc >= 1.0 - 1e-9 and action != 0),
            energy_from_grid_kwh=energy_from_grid_kwh,
            energy_to_battery_kwh=energy_to_battery_kwh,
        )

        self.soc = new_soc

        return energy_from_grid_kwh, meta

    def cost_eur(self, energy_from_grid_kwh: float, price_eur_mwh: float) -> float:
        """calculates how much money you pay for the electricity you took from the grid.
        Cost [EUR] = E_grid [kWh] * p [EUR/MWh] / 1000 (we have to divide by 100 in order to get a value in kWh rather than MWh)"""
        return energy_from_grid_kwh * price_eur_mwh / 1000.0

    def shortfall_kwh(self, target_soc: float) -> float:
        """How much energy is still missing before the battery reaches the target SoC?"""
        return max(0.0, target_soc - self.soc) * self.battery_capacity_kwh #(missing percentage multiplied by battery capacity)


def rollout(actions: Sequence[int], prices: Sequence[float], initial_soc: float = 0.3,
            battery_capacity_kwh: float = 60.0, charging_efficiency: float = 0.90):
    """Runs the battery model for a sequence of (action, price) pairs (one
    overnight episode)"""
    model = EVBatteryModel(battery_capacity_kwh, charging_efficiency, initial_soc)

    soc_trace, cost_trace, capped_trace = [], [], []
    for action, price in zip(actions, prices):
        energy_kwh, meta = model.step(action)
        cost = model.cost_eur(energy_kwh, price)
        soc_trace.append(model.soc)
        cost_trace.append(cost)
        capped_trace.append(meta["capped"])

    return soc_trace, cost_trace, capped_trace

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    demo_prices = 50 + 30 * np.sin(np.linspace(0, 3.14, 14)) + rng.normal(0, 5, 14)
    demo_hours = list(range(17, 24)) + list(range(0, 7))

    actions_full = [2] * 14
    soc_trace, cost_trace, capped_trace = rollout(actions_full, demo_prices, initial_soc=0.2)
    print("Always-full: final SoC", soc_trace[-1], "total cost", sum(cost_trace))
