from __future__ import annotations
from typing import Any, Tuple

import numpy as np
import matplotlib.pyplot as plt

from environment.discrete_env import make_train_val_envs
from agent.baseline import NaiveImmediateChargePolicy, TrainedQPolicy
from agent.rollout_env import rollout, RolloutSignals
from agent.q_learning_agent import QLearning

ACTION_NAMES = {0: "idle", 1: "half (50%)", 2: "full (100%)"}
ACTION_POWER_KW = {0: 0.0, 1: 5.5, 2: 11.0}
ACTION_COLORS = ["#dddddd", "#f6a04d", "#d1362f"]

HOUR_ORDER = list(range(17, 24)) + list(range(0, 7))


def _reindex_by_hour_order(hours: np.ndarray, values: np.ndarray, agg="mean"):
    """Aggregate `values` by hour-of-day and return them in HOUR_ORDER."""
    out = []
    for h in HOUR_ORDER:
        mask = hours == h
        if mask.sum() == 0:
            out.append(np.nan)
        else:
            out.append(values[mask].mean() if agg == "mean" else values[mask].sum())
    return np.array(out)


# "In which hour does it charge, and how much?" 
def plot_action_by_hour(sig: RolloutSignals, title: str):
    n_hours = len(HOUR_ORDER)
    shares = np.zeros((n_hours, 3))
    mean_price = np.full(n_hours, np.nan)

    for i, h in enumerate(HOUR_ORDER):
        mask = sig.hour_of_day == h
        if mask.sum() == 0:
            continue
        for a in (0, 1, 2):
            shares[i, a] = (sig.action[mask] == a).mean()
        mean_price[i] = sig.price[mask].mean()

    fig, ax = plt.subplots(figsize=(11, 5))
    bottom = np.zeros(n_hours)
    x = np.arange(n_hours)
    for a in (0, 1, 2):
        ax.bar(x, shares[:, a], bottom=bottom, color=ACTION_COLORS[a], label=ACTION_NAMES[a], width=0.85)
        bottom += shares[:, a]

    ax.set_xticks(x)
    ax.set_xticklabels([f"{h:02d}:00" for h in HOUR_ORDER], rotation=45)
    ax.set_ylabel("share of nights")
    ax.set_xlabel("hour of day (arrival window \u2192 07:00 deadline)")
    ax.set_title(title)

    ax2 = ax.twinx()
    ax2.plot(x, mean_price, color="black", marker="o", ms=4, lw=1.5, label="mean price (EUR/MWh)")
    ax2.set_ylabel("mean price (EUR/MWh)")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper center", ncol=4, fontsize=8)

    plt.tight_layout()
    plt.show()
    return fig, ax


# Agent vs naive: average charging power drawn, by hour of day
def plot_power_by_hour_comparison(agent_sig: RolloutSignals, naive_sig: RolloutSignals, title: str):
    agent_power = np.array([ACTION_POWER_KW[a] for a in agent_sig.action])
    naive_power = np.array([ACTION_POWER_KW[a] for a in naive_sig.action])

    agent_by_hour = _reindex_by_hour_order(agent_sig.hour_of_day, agent_power)
    naive_by_hour = _reindex_by_hour_order(naive_sig.hour_of_day, naive_power)
    price_by_hour = _reindex_by_hour_order(agent_sig.price, agent_sig.price)

    n_hours = len(HOUR_ORDER)
    x = np.arange(n_hours)

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.plot(x, naive_by_hour, color="tab:blue", marker="o", label="naive — mean charging power (kW)")
    ax1.plot(x, agent_by_hour, color="tab:green", marker="o", label="Q-learning — mean charging power (kW)")
    ax1.set_ylabel("mean charging power (kW)")
    ax1.set_xlabel("hour of day (arrival window \u2192 07:00 deadline)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{h:02d}:00" for h in HOUR_ORDER], rotation=45)

    ax2 = ax1.twinx()
    ax2.bar(x, price_by_hour, color="grey", alpha=0.2, label="mean price (EUR/MWh)")
    ax2.set_ylabel("mean price (EUR/MWh)")
    ax2.set_zorder(0)
    ax1.set_zorder(1)
    ax1.patch.set_visible(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", fontsize=8)
    ax1.set_title(title)

    plt.tight_layout()
    plt.show()
    return fig, ax1


# Price paid, conditional on action (agent only) 
def plot_price_by_action(sig: RolloutSignals, title: str):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    data = [sig.price[sig.action == a] for a in (0, 1, 2)]
    bp = ax.boxplot(data, tick_labels=[ACTION_NAMES[a] for a in (0, 1, 2)], patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], ACTION_COLORS):
        patch.set_facecolor(color)
    ax.set_ylabel("price (EUR/MWh)")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
    return fig, ax


def plot_cumulative_cost(
    *,
    sigs: dict,  # {label: RolloutSignals}
    title: str = "Cumulative cost over time",
) -> Tuple[Any, Any]:
    """
    Each policy's own cumulative SPEND over time, in positive EUR (cost =
    -reward) (so "up" here means
    "spent more") Plotting multiple
    policies on the same axes makes the savings visible directly as the
    widening gap between their cost curves, not just as a derived delta.
    """
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for label, sig in sigs.items():
        cumulative_cost = -np.cumsum(sig.episode_total_reward)
        ax.plot(np.arange(len(cumulative_cost)), cumulative_cost, label=label)
    ax.set_xlabel("episode (night) index, chronological")
    ax.set_ylabel("cumulative cost (EUR)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()
    return fig, ax

def main():
    PRICE_N_BINS = 3
    SOC_N_BINS = 10
    train_env, val_env = make_train_val_envs(
        "./data/Germany.csv", price_n_bins=PRICE_N_BINS, soc_n_bins=SOC_N_BINS
    )

    try:
        agent, _ = QLearning.load("q_agent_checkpoint")
        print("Loaded trained agent from q_agent_checkpoint/.")
    except FileNotFoundError:
        print("No checkpoint found — training a fresh agent (see train_agent.py)...")
        import train_agent as tq
        agent, _, _, _ = tq.train(train_env, tq.N_TRAINING_EPISODES, seed=tq.SEED)

    agent_policy = TrainedQPolicy(agent)
    naive_policy = NaiveImmediateChargePolicy(train_env.quantizers["soc"], train_env.target_soc)

    train_windows = train_env.loader.chronological_episodes("train", np.random.default_rng(42))
    val_windows = val_env.loader.chronological_episodes("val", np.random.default_rng(42))

    sig_agent_train = rollout(env=train_env, policy=agent_policy, episode_windows=train_windows, seed=42)
    sig_naive_train = rollout(env=train_env, policy=naive_policy, episode_windows=train_windows, seed=42)
    sig_agent_val = rollout(env=val_env, policy=agent_policy, episode_windows=val_windows, seed=42)
    sig_naive_val = rollout(env=val_env, policy=naive_policy, episode_windows=val_windows, seed=42)

    plot_action_by_hour(sig_agent_train, title="Q-learning: charging behavior by hour of day (train)")
    plot_action_by_hour(sig_naive_train, title="Naive: charging behavior by hour of day (train), for comparison")

    plot_power_by_hour_comparison(sig_agent_train, sig_naive_train,
                                   title="Average charging power by hour: Q-learning vs naive (train)")

    plot_price_by_action(sig_agent_train, title="Q-learning: price paid, conditional on action (train)")

    plot_cumulative_cost(
        sigs={
            "Q-learning (val)": sig_agent_val,
            "Naive (val)": sig_naive_val,
        },
        title="Cumulative cost over time: Q-learning vs naive",
    )


if __name__ == "__main__":
    main()