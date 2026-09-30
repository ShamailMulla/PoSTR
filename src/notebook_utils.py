"""
Shared utilities for BTRL transition network notebooks.
"""
from copy import deepcopy
from bsuite.environments.deep_sea import DeepSea
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from TRL.common.utils import init_env, env_reset, env_step, compute_state_loss


def make_env(size, deterministic=True, seed=0):
    """Thin wrapper around bsuite DeepSea — avoids gym_wrapper (requires gym, not installed)."""
    env = DeepSea(size, deterministic=deterministic, randomize_actions=False, seed=seed)
    return env

def ds_reset(env):
    return env.reset().observation.flatten().astype(np.float32)

def ds_step(env, action):
    ts = env.step(int(action))
    obs  = ts.observation.flatten().astype(np.float32)
    rew  = float(ts.reward) if ts.reward is not None else 0.0
    done = ts.last()
    return obs, rew, bool(done)


def generate_transition_data(config_or_size, num_transitions_or_ctx=1000, num_transitions=None):
    """
    Generate training transitions and a per-state test set.

    Calling conventions:
        generate_transition_data(config, num_transitions=1000)
            config must have config['experiment']['env'] (int) and
            config['transition']['context_length'] (int).
        generate_transition_data(env_size, ctx_len, num_transitions=1000)
            Legacy integer form used by value network notebooks.

    Returns:
        episodes          — dict of training transitions
        test_transitions  — dict[state_idx] → {trajectory, timestep, next_state}
        obs_space         — env_size * env_size
        possible_actions  — 1-D int tensor of action indices
    """
    if isinstance(config_or_size, int):
        env_size = config_or_size
        ctx_len  = int(num_transitions_or_ctx)
        n_trans  = int(num_transitions) if num_transitions is not None else 1000
    else:
        cfg      = config_or_size
        env_size = int(cfg['experiment']['env'])
        ctx_len  = int(cfg['transition']['context_length'])
        n_trans  = int(num_transitions_or_ctx)

    env       = make_env(env_size, deterministic=True, seed=0)
    obs_space = env_size * env_size
    n_actions = env.action_spec().num_values
    possible_actions = torch.arange(n_actions)

    state = ds_reset(env)

    episodes = {
        'states':      [None] * n_trans,
        'next_states': [],
        'actions':     [],
        'rewards':     [],
        'terminals':   [],
        'timestep':    [None] * n_trans,
    }

    timestep     = 0
    trajectory   = np.zeros((ctx_len, obs_space)).tolist()
    ts_buf       = np.zeros(ctx_len, dtype=int).tolist()
    state_to_ctx = {}   # state_idx → (trajectory, ts_buf) — last seen context
    sa_to_next   = {}   # (state_idx, action) → next_state_idx

    for i in range(n_trans):
        action     = int(np.random.randint(0, n_actions))
        next_state, reward, done = ds_step(env, action)

        trajectory.append(state.tolist()); trajectory.pop(0)
        ts_buf.append(timestep);           ts_buf.pop(0)

        state_idx = int(np.argmax(state))
        ns_idx    = int(np.argmax(next_state))

        episodes['states'][i]   = deepcopy(trajectory)
        episodes['timestep'][i] = deepcopy(ts_buf)
        episodes['next_states'].append(next_state.tolist())
        episodes['actions'].append(action)
        episodes['rewards'].append(reward)
        episodes['terminals'].append(done)

        state_to_ctx[state_idx]         = (deepcopy(trajectory), deepcopy(ts_buf))
        sa_to_next[(state_idx, action)] = ns_idx

        if done:
            state      = ds_reset(env)
            trajectory = np.zeros((ctx_len, obs_space)).tolist()
            timestep   = 0
            ts_buf     = np.zeros(ctx_len, dtype=int).tolist()
        else:
            state     = next_state
            timestep += 1

    # Build per-state test set from observed (state, action) → next_state pairs
    test_transitions = {}
    for state_idx, (traj, ts) in state_to_ctx.items():
        test_transitions[state_idx] = {
            'trajectory': traj,
            'timestep':   ts,
            'next_state': {a: sa_to_next.get((state_idx, a), None) for a in range(n_actions)},
        }

    return episodes, test_transitions, obs_space, possible_actions


def _run_test(model, test_transitions, possible_actions, n_iterations, lock_per_iter):
    """
    Shared test loop used by both locked and unlocked evaluation functions.

    lock_per_iter=True  — sample a fresh world hypothesis M̂ via lock_all_dropouts
                          before each iteration, unlock after.  Variance across
                          iterations = inter-hypothesis (Bayesian) uncertainty.
    lock_per_iter=False — no locking; each predict() draws a fresh stochastic mask.
                          Demonstrates the pre-fix non-stationary (Dithering Trap)
                          behaviour.

    Grid metrics
    ────────────
    DeepSea states are flat indices into an NxN grid (state s → row=s//N, col=s%N).
    Flat-index distance is misleading at row boundaries (e.g. indices 4 and 5 have
    flat-diff=1 but grid-distance≈4.1 in a 5×5 grid).  We therefore return:

    mean_grid_dist  — mean Euclidean distance in (row,col) space between each
                      sampled hypothesis prediction and the true next state.
                      Averages over all n_iterations for this (state, action) pair.
    grid_variance   — var(pred_rows) + var(pred_cols) across the n_iterations.
                      Measures posterior spread in grid space; unaffected by the
                      row-boundary artefact of flat-index variance.
    reward_mse      — (mean predicted reward − true reward)² for this (state, action)
                      pair.  True reward = 1 iff the true next state is the goal
                      (obs_space − 1), else 0.  Measures reward-head quality.
    variance        — kept for backward compatibility with the test notebooks
                      (flat-index variance of the predictions).
    """
    obs_space      = len(list(test_transitions.values())[0]['trajectory'][0])
    N              = int(round(obs_space ** 0.5))   # grid side length (e.g. 5 for DeepSea-5)
    test_state_ids = list(test_transitions.keys())
    n_actions      = len(possible_actions)
    ctx            = len(list(test_transitions.values())[0]['trajectory'])
    device         = next(model.parameters()).device

    possible_actions = possible_actions.to(device)

    pred     = {s: {a: [] for a in range(n_actions)} for s in test_state_ids}
    rew_pred = {s: {a: [] for a in range(n_actions)} for s in test_state_ids}

    for _ in range(n_iterations):
        if lock_per_iter:
            model.lock_all_dropouts(n_actions, ctx, device)

        for state_idx in test_state_ids:
            obs = torch.tensor(
                test_transitions[state_idx]['trajectory'], dtype=torch.float32
            ).unsqueeze(0)
            obs = torch.cat([obs] * n_actions, dim=0).to(device)

            t  = torch.tensor(test_transitions[state_idx]['timestep'], dtype=int).unsqueeze(0)
            ts = torch.cat([t] * n_actions, dim=0).to(device)

            preds, rews = model.predict(ts, obs, possible_actions)
            preds = torch.argmax(preds, dim=1).squeeze().cpu()
            rews  = rews.squeeze(-1).cpu()   # (n_actions,)

            for a_idx in range(n_actions):
                pred[state_idx][a_idx].append(preds[a_idx].item())
                rew_pred[state_idx][a_idx].append(rews[a_idx].item())

        if lock_per_iter:
            model.unlock_all_dropouts()

    true_next = {s: test_transitions[s]['next_state'] for s in test_state_ids}
    rows = []
    for state_idx in test_state_ids:
        for a_idx, a_val in enumerate(possible_actions.cpu()):
            ns = true_next[state_idx][a_idx]
            if ns is None:
                ns = obs_space - 1

            p_list         = pred[state_idx][a_idx]
            r_list         = rew_pred[state_idx][a_idx]
            consensus_pred = int(np.round(np.mean(p_list)))

            # Grid-space metrics
            true_row, true_col = divmod(int(ns), N)
            pred_rows = [int(p) // N for p in p_list]
            pred_cols = [int(p) % N  for p in p_list]
            grid_dists = [
                np.sqrt((pr - true_row) ** 2 + (pc - true_col) ** 2)
                for pr, pc in zip(pred_rows, pred_cols)
            ]

            # Reward-head metrics: true reward = 1 iff next state is the goal
            true_reward      = 1.0 if int(ns) == obs_space - 1 else 0.0
            mean_pred_reward = float(np.mean(r_list))
            reward_mse       = (mean_pred_reward - true_reward) ** 2

            rows.append({
                'current_state':   state_idx,
                'action':          a_val.item(),
                'true next_state': ns,
                'true_row':        true_row,
                'true_col':        true_col,
                'pred next_state': consensus_pred,
                'pred_row':        int(np.round(np.mean(pred_rows))),
                'pred_col':        int(np.round(np.mean(pred_cols))),
                'mean_grid_dist':  np.round(np.mean(grid_dists), 4),
                'grid_variance':   np.round(np.var(pred_rows) + np.var(pred_cols), 4),
                'exact_match':     int(consensus_pred == ns),
                'pred_reward':     np.round(mean_pred_reward, 4),
                'true_reward':     true_reward,
                'reward_mse':      np.round(reward_mse, 6),
                'variance':        np.round(np.var(p_list), 4),  # flat-index, kept for compat
            })
    return pd.DataFrame(rows)


def test_model_locked(model, test_transitions, possible_actions, config=None, n_hypotheses=100):
    """
    Evaluate by sampling n_hypotheses different world hypotheses M̂.

    Each iteration: lock_all_dropouts → predict all test cases → unlock_all_dropouts.
    Variance = inter-hypothesis disagreement — the correct Bayesian uncertainty
    metric for a Thompson-Sampling agent.  For a well-trained model on a
    deterministic environment this should approach 0.
    """
    return _run_test(model, test_transitions, possible_actions, n_hypotheses, lock_per_iter=True)


def test_model_unlocked(model, test_transitions, possible_actions, config=None, n_calls=100):
    """
    Evaluate without locking masks (pre-fix / broken behaviour).

    Each predict() call draws a fresh random dropout mask, so the same
    (state, action) pair produces different predictions every call.
    Variance > 0 for a deterministic environment = the Dithering Trap bug.
    """
    return _run_test(model, test_transitions, possible_actions, n_calls, lock_per_iter=False)


def early_stopping_train(model, states, next_states, timesteps, actions, rewards,
                          optimizer, obs_type=None, obs_space=None,
                          max_epochs=300, patience=15, min_delta=5e-4, device=None):
    """
    Train with environment-aware state loss and early stopping.

    obs_type and obs_space are forwarded to compute_state_loss() from TRL.common.utils.
    If not provided, obs_type defaults to model.obs_type and obs_space to model.states.

    device: if None, inferred from model parameters. All data tensors are moved to
            this device before training begins.

    Returns:
        loss_history:       list of per-epoch total loss values
        epochs_to_converge: epoch at which early stopping triggered (max_epochs if never)
        best_loss:          lowest training loss achieved
    """
    if obs_type is None:
        obs_type = getattr(model, 'obs_type', 'grid')
    if obs_space is None:
        obs_space = getattr(model, 'states', next_states.shape[-1])
    if device is None:
        device = next(model.parameters()).device

    states      = states.to(device)
    next_states = next_states.to(device)
    timesteps   = timesteps.to(device)
    actions     = actions.to(device)
    rewards     = rewards.to(device)

    best_loss          = float('inf')
    patience_counter   = 0
    loss_history       = []
    epochs_to_converge = max_epochs

    for epoch in range(max_epochs):
        pred_states, pred_rewards, _ = model.forward(timesteps, states, actions)
        state_loss  = compute_state_loss(
            pred_states, next_states.squeeze(1),
            obs_type=obs_type, obs_space=obs_space,
        )
        reward_loss = F.mse_loss(pred_rewards, rewards)
        loss        = state_loss + reward_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if loss_val < best_loss - min_delta:
            best_loss        = loss_val
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            epochs_to_converge = epoch + 1
            break

    return loss_history, epochs_to_converge, best_loss
