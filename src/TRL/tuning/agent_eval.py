"""
Full-agent evaluation harness for hyperparameter validation and online tuning.

Replicates the interaction loop of main.py (same select_action/update calling
convention) but without TensorBoard, checkpointing, or log directories, and
returns sample-efficiency metrics instead of printing.

Metric definitions (DeepSea):
  Every episode lasts exactly N steps (the agent descends one row per step),
  so within-episode steps-to-goal is a constant and cannot be optimised. What
  varies is how much training experience the agent needs before it reliably
  reaches the goal — sample efficiency — captured here as cumulative regret
  and episodes_to_solve.

  optimal_return     — 0.99 (goal reward, see env_step) minus the per-step
                       move costs along the rightward path.
  episode regret     — optimal_return - episode_return.
  cumulative_regret  — sum of episode regrets over the training budget. Flat
                       slope = solved; the earlier the curve bends, the more
                       sample-efficient the agent. Primary online objective.
  first_goal_episode — first episode in which the goal state was reached.
  episodes_to_solve  — first episode where the trailing SOLVE_WINDOW mean
                       return >= SOLVE_FRACTION * optimal_return (censored at
                       the total episode count if never reached).
  solved_rate_last_100 — fraction of goal-reaching episodes in the last 100.
"""
import io
from contextlib import redirect_stdout

import numpy as np

from ..agent.btrl import BTRL
from ..common.logger import Logger
from ..common.utils import init_env, env_reset, env_step
from .common import set_global_seed

SOLVE_WINDOW = 20
SOLVE_FRACTION = 0.9
GOAL_REWARD_THRESHOLD = 0.5  # non-goal episode returns are <= 0


class NullDataManager:
    """Logger backend that discards scalars and never touches the filesystem."""

    def update(self, log: dict, timestep: int):
        pass

    def save(self, agent, timestep: int):
        pass


def optimal_return(env_size: int) -> float:
    # env_step overrides the goal reward to 0.99; each rightward move on the
    # optimal path costs 0.01/N (bsuite DeepSea), paid on the N-1 pre-goal steps.
    return 0.99 - 0.01 * (env_size - 1) / env_size


def run_agent_trial(config: dict, seed: int, total_steps: int, quiet: bool = True) -> dict:
    """
    Train one BTRL agent from scratch for total_steps environment steps.
    Returns per-episode returns and the sample-efficiency metrics above.
    """
    if float(config["algorithm"]["policy_noise"]) != 0.0:
        # BTRL.select_action's epsilon branch returns a bare action (no
        # trajectory/context), which the main-loop calling convention cannot
        # unpack. Thompson Sampling needs no dithering noise anyway.
        raise ValueError("run_agent_trial requires algorithm.policy_noise == 0.0")

    set_global_seed(seed)
    env_size = int(config["experiment"]["env"])
    time_limit = config["experiment"]["time_limit"]
    suite = config["experiment"]["suite"]

    sink = io.StringIO() if quiet else None

    def _run():
        env, actions, _, obs_space = init_env(
            suite, config["experiment"]["env"], test=False,
            deterministic=config["experiment"]["deterministic"],
            randomize_actions=config["experiment"].get("randomize_actions", True),
            mapping_seed=config["experiment"].get("mapping_seed", seed),
            legacy_goal=config["experiment"].get("legacy_goal", False),
        )
        agent = BTRL(config, actions, Logger(NullDataManager()), obs_space, seed)

        episode_returns = []
        experiment_step = 1
        ep = 0
        while experiment_step <= total_steps:
            episode_step = 1
            episode_return = 0.0
            current_observation = env_reset(env, suite)
            done = False
            while not done:
                action, trajectory, time_transition = agent.select_action(
                    current_observation, episode_step
                )
                next_state, reward, done = env_step(env, suite, action)
                done = done or episode_step == time_limit
                agent.update(
                    trajectory, action, reward, next_state, done,
                    ep, experiment_step, time_transition,
                )
                episode_return += reward
                current_observation = next_state
                episode_step += 1
                experiment_step += 1
            episode_returns.append(episode_return)
            ep += 1
        return episode_returns

    if quiet:
        with redirect_stdout(sink):
            episode_returns = _run()
    else:
        episode_returns = _run()

    returns = np.array(episode_returns)
    n_episodes = len(returns)
    opt = optimal_return(env_size)

    regrets = opt - returns
    goal_reached = returns > GOAL_REWARD_THRESHOLD

    first_goal_episode = int(np.argmax(goal_reached)) if goal_reached.any() else n_episodes

    episodes_to_solve = n_episodes  # censored value if never solved
    if n_episodes >= SOLVE_WINDOW:
        trailing = np.convolve(returns, np.ones(SOLVE_WINDOW) / SOLVE_WINDOW, mode="valid")
        solved = trailing >= SOLVE_FRACTION * opt
        if solved.any():
            episodes_to_solve = int(np.argmax(solved)) + SOLVE_WINDOW

    last = goal_reached[-100:]
    return {
        "seed": seed,
        "episode_returns": episode_returns,
        "cumulative_regret": float(regrets.sum()),
        "first_goal_episode": first_goal_episode,
        "episodes_to_solve": episodes_to_solve,
        "solved": bool(episodes_to_solve < n_episodes),
        "solved_rate_last_100": float(last.mean()) if len(last) else 0.0,
        "n_episodes": n_episodes,
        "optimal_return": opt,
    }


def evaluate_config(config: dict, seeds, total_steps: int, quiet: bool = True) -> dict:
    """Run one config across several seeds and aggregate the metrics."""
    per_seed = [run_agent_trial(config, seed, total_steps, quiet=quiet) for seed in seeds]

    def agg(key):
        values = np.array([r[key] for r in per_seed], dtype=float)
        return float(values.mean()), float(values.std())

    summary = {"per_seed": per_seed, "seeds": list(seeds), "total_steps": total_steps}
    for key in ["cumulative_regret", "first_goal_episode", "episodes_to_solve", "solved_rate_last_100"]:
        summary[f"{key}_mean"], summary[f"{key}_std"] = agg(key)
    summary["solved_fraction"] = float(np.mean([r["solved"] for r in per_seed]))
    return summary
