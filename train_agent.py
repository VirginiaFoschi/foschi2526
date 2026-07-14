"""
Train the tabular Q-learning agent on EVChargingEnvDiscrete, then evaluate
the greedy policy against the naive baseline — chronologically,
on both train and validation — to see whether learning actually beat them.
"""

import time
from collections import deque
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from environment.discrete_env import make_train_val_envs
from agent.baseline import NaiveImmediateChargePolicy, TrainedQPolicy
from agent.rollout_env import rollout
from agent.q_learning_agent import QLearning
from agent.utils import Experience, LinearSchedule


# ----------------------------------------------------------------------
# Hyperparameters
# ----------------------------------------------------------------------
N_TRAINING_EPISODES = 30_000    # random nights sampled (with repetition) from train split
DISCOUNT = 0.95                  # gamma
LEARNING_RATE_START = 0.10       # alpha — decays to LEARNING_RATE_END if DECAY_LEARNING_RATE=True
LEARNING_RATE_END = 0.02
DECAY_LEARNING_RATE = True       # constant alpha never lets the Q-table settle; see conversation
EPS_START, EPS_END = 1.0, 0.05   # epsilon-greedy schedule
EPS_DECAY_FRACTION = 0.8         # decay eps to EPS_END over this fraction of total env steps
QVAL_INIT = 0.0
SEED = 0

PRICE_N_BINS = 3
SOC_N_BINS = 10
MAX_HOURS_UNTIL_DEPARTURE = 14   # arrival as early as 17:00 -> departure 07:00 = 14h

EVAL_EVERY_N_EPISODES = 2000     # how often to evaluate the FROZEN greedy policy during training


def train(train_env, n_episodes: int, seed: int = SEED, val_env=None, val_windows=None):
    train_env.reset(seed=seed)

    # rough estimate of total env steps, to size the epsilon/alpha decay schedules
    approx_steps_per_episode = 12.5
    total_steps_estimate = int(n_episodes * approx_steps_per_episode)

    eps_schedule = LinearSchedule(
        start=EPS_START, end=EPS_END,
        n_steps=int(total_steps_estimate * EPS_DECAY_FRACTION),
    )

    if DECAY_LEARNING_RATE:
        alpha = LinearSchedule(start=LEARNING_RATE_START, end=LEARNING_RATE_END, n_steps=total_steps_estimate)
    else:
        alpha = LEARNING_RATE_START

    agent = QLearning(
        obs_space_dims=3,          # (soc_bin, hours_until_departure, price_bin) — see reduced_tuple()
        action_space_dims=3,       # idle / half / full
        update_coefficient=alpha,
        discount=DISCOUNT,
        eps=eps_schedule,
        qval_init=QVAL_INIT,
        seed=seed,
    )

    episode_rewards = np.zeros(n_episodes, dtype=float)
    window = deque(maxlen=100)
    smoothed_rewards = np.zeros(n_episodes, dtype=float)

    # If I froze the agent's current knowledge and made
    # it always pick its best guess (no randomness at all 
    # because otherwise with epsilon-greedy the agent would act partly randomly)
    # how would it do on the exact same fixed set of validation nights, 
    # every single time I check?
    eval_episode_idx: List[int] = []
    eval_mean_reward: List[float] = []

    def evaluate_greedy(env, windows, policy):
        sig = rollout(env=env, policy=policy, episode_windows=windows, seed=999)
        return sig.episode_total_reward.mean()

    t0 = time.time()
    for ep in range(n_episodes):
        obs, info = train_env.reset()
        agent.reset()

        terminated = truncated = False
        ep_reward = 0.0
        while not (terminated or truncated):
            a = agent.act(obs)
            next_obs, r, terminated, truncated, info = train_env.step(a)
            done = terminated or truncated

            agent.step(Experience(s=obs, a=a, r=r, sp=next_obs, done=done))

            obs = next_obs
            ep_reward += r

        episode_rewards[ep] = ep_reward
        window.append(ep_reward)
        smoothed_rewards[ep] = np.mean(window)

        if val_env is not None and val_windows is not None and (ep + 1) % EVAL_EVERY_N_EPISODES == 0:
            greedy_mean = evaluate_greedy(val_env, val_windows, TrainedQPolicy(agent))
            eval_episode_idx.append(ep + 1)
            eval_mean_reward.append(greedy_mean)

        if (ep + 1) % 5000 == 0:
            elapsed = time.time() - t0
            eps_now = agent.eps.value if hasattr(agent.eps, "value") else agent.eps
            alpha_now = agent.update_coefficient.value if hasattr(agent.update_coefficient, "value") else agent.update_coefficient
            print(f"  episode {ep + 1:>6}/{n_episodes}  "
                  f"smoothed_reward(last100, eps-greedy)={smoothed_rewards[ep]:7.3f}  "
                  f"eps={eps_now:.3f}  alpha={alpha_now:.3f}  "
                  f"|Q|={len(agent.Q)} states visited  "
                  f"({elapsed:.1f}s elapsed)")

    print(f"Training done in {time.time() - t0:.1f}s. "
          f"{len(agent.Q)} distinct states visited "
          f"(out of up to {SOC_N_BINS * MAX_HOURS_UNTIL_DEPARTURE * PRICE_N_BINS} possible "
          f"in the reduced (soc, hours_left, price) state space).")

    eval_curve = (np.array(eval_episode_idx), np.array(eval_mean_reward)) if eval_episode_idx else None
    return agent, episode_rewards, smoothed_rewards, eval_curve


def plot_learning_curve(smoothed_rewards: np.ndarray, eval_curve=None, title: str = "Q-learning training curve"):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(np.arange(len(smoothed_rewards)), smoothed_rewards, alpha=0.6,
            label="epsilon-greedy training reward (100-ep rolling mean) — noisy")
    if eval_curve is not None:
        idx, vals = eval_curve
        ax.plot(idx, vals, marker="o", color="tab:red", lw=2,
                label="FROZEN greedy policy, evaluated on validation — the reliable signal")
    ax.set_xlabel("training episode")
    ax.set_ylabel("reward per episode (EUR)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.show()
    return fig, ax


def main():
    train_env, val_env = make_train_val_envs(
        "./data/Germany.csv",
        price_n_bins=PRICE_N_BINS,
        soc_n_bins=SOC_N_BINS,
    )

    val_windows_for_eval = val_env.loader.chronological_episodes("val", np.random.default_rng(123))

    print(f"Training tabular Q-learning for {N_TRAINING_EPISODES} episodes...")
    agent, episode_rewards, smoothed_rewards, eval_curve = train(
        train_env, N_TRAINING_EPISODES, seed=SEED,
        val_env=val_env, val_windows=val_windows_for_eval,
    )

    plot_learning_curve(smoothed_rewards, eval_curve=eval_curve,
                         title="Q-learning training curve (noisy training reward vs. clean greedy eval)")

    naive_policy = NaiveImmediateChargePolicy(train_env.quantizers["soc"], train_env.target_soc)
    trained_policy = TrainedQPolicy(agent)

    train_windows = train_env.loader.chronological_episodes("train", np.random.default_rng(42))
    val_windows = val_env.loader.chronological_episodes("val", np.random.default_rng(42))

    results = {}
    for split_name, env, windows in [("train", train_env, train_windows), ("val", val_env, val_windows)]:
        for policy_name, policy in [("naive", naive_policy), ("qlearning", trained_policy)]:
            sig = rollout(env=env, policy=policy, episode_windows=windows, seed=42)
            results[(policy_name, split_name)] = sig
            print(f"{policy_name:>10} / {split_name:<5}: mean reward/night = "
                  f"{sig.episode_total_reward.mean():7.3f} EUR, "
                  f"total = {sig.episode_total_reward.sum():9.1f} EUR")

    print("\n\n")
    print("SUMMARY: mean reward per night (higher / less negative = better)")
    header = f"{'policy':<12}{'train':>12}{'validation':>12}"
    print(header)
    for p in ("naive", "qlearning"):
        row = f"{p:<12}{results[(p,'train')].episode_total_reward.mean():>12.3f}" \
              f"{results[(p,'val')].episode_total_reward.mean():>12.3f}"
        print(row)


    # Save the trained agent
    QLearning.save(
        agent, path="q_agent_checkpoint",
        envconstructkwargs=dict(price_n_bins=PRICE_N_BINS, soc_n_bins=SOC_N_BINS),
    )

    return agent, results


if __name__ == "__main__":
    main()
