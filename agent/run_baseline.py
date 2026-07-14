from agent.baseline import NaiveImmediateChargePolicy
from agent.rollout_env import rollout
from environment.discrete_env import make_train_val_envs
from plot_agent import plot_action_by_hour, plot_cumulative_cost
import numpy as np

def main():
    train_env, val_env = make_train_val_envs(
        "data/Germany.csv",
        price_n_bins=3,
        soc_n_bins=10,
    )

    naive_policy = NaiveImmediateChargePolicy(
        soc_quantizer=train_env.quantizers["soc"],
        target_soc=train_env.target_soc,
    )

    train_windows = train_env.loader.chronological_episodes("train", np.random.default_rng(42))
    val_windows = val_env.loader.chronological_episodes("val", np.random.default_rng(42))

    print(f"Rolling out both policies over {len(train_windows)} train nights "
          f"and {len(val_windows)} validation nights, in chronological order...")

    sig_naive_train = rollout(env=train_env, policy=naive_policy, episode_windows=train_windows, seed=42)

    sig_naive_val = rollout(env=val_env, policy=naive_policy, episode_windows=val_windows, seed=42)

    for label, sig in [
        ("Naive   / train", sig_naive_train),
        ("Naive   / val", sig_naive_val),
    ]:
        print(f"{label}: mean reward/night = {sig.episode_total_reward.mean():7.3f} EUR, "
              f"total over period = {sig.episode_total_reward.sum():9.1f} EUR")

    plot_action_by_hour(sig_naive_train, title="Naive policy: charging behavior by hour of day (train)")
 
    # HOW MUCH does it cost, over time?
    plot_cumulative_cost(
        sigs={"train (2022-2024)": sig_naive_train, "validation (2025)": sig_naive_val},
        title="Naive policy: cumulative cost over time",
    )


if __name__ == "__main__":
    main()