"""
Full-state checkpointing for exact resumption of a BTRL run.

DataManager.save() writes only network weights + the main replay ring, which is
enough for evaluation but NOT for continuing training: it omits the protected
replay ring, the optimizer moments, the value target network, every RNG stream,
and the step/episode counters. Restoring from it would restart training at step
1 with re-initialised optimisers and a lost protected buffer.

save_checkpoint / load_checkpoint capture the complete agent + loop state in a
single file so a killed run can be resumed exactly where it stopped.
"""
from copy import deepcopy

import numpy as np
import torch


def save_checkpoint(agent, experiment_step: int, ep: int, path: str):
    tn = agent.model.transition_network
    en = agent.model.terminal_network
    vn = agent.value_network
    ds = agent.dataset
    pt = agent.policy_trainer

    ckpt = {
        # networks
        "transition": tn.state_dict(),
        "terminal": en.state_dict(),
        "value": vn.state_dict(),
        # optimiser moments (Adam state) — needed for smooth continuation
        "transition_opt": tn.optimizer.state_dict(),
        "terminal_opt": en.optimizer.state_dict(),
        "value_opt": vn.optimizer.state_dict(),
        # value target network (deepcopy of value layers; None before first update)
        "target_net": pt.target_net.state_dict() if pt.target_net is not None else None,
        # replay buffer: BOTH rings + all bookkeeping counters
        "dataset": {
            "episodes": ds.episodes,
            "goal_episodes": ds.goal_episodes,
            "cum_rew": ds.cum_rew,
            "add_idx": ds.add_idx,
            "n_samples": ds.n_samples,
            "max_ep_len": ds.max_ep_len,
            "min_ep_len": ds.min_ep_len,
            "episode_add_idx": ds.episode_add_idx,
            "total_num_transitions": ds.total_num_transitions,
            "replace": ds.replace,
            "goal_add_idx": ds.goal_add_idx,
            "goal_n_samples": ds.goal_n_samples,
            "goal_replace": ds.goal_replace,
            "rng": ds.random_state.get_state(),
        },
        # RNG streams
        "agent_rng": agent.random_state.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng": np.random.get_state(),
        # loop counters
        "experiment_step": experiment_step,
        "ep": ep,
    }
    torch.save(ckpt, path)


def load_checkpoint(agent, path: str):
    """Restore full state into a freshly built agent. Returns (experiment_step, ep)."""
    device = agent.device
    ckpt = torch.load(path, map_location=device, weights_only=False)

    tn = agent.model.transition_network
    en = agent.model.terminal_network
    vn = agent.value_network
    ds = agent.dataset
    pt = agent.policy_trainer

    tn.load_state_dict(ckpt["transition"])
    en.load_state_dict(ckpt["terminal"])
    vn.load_state_dict(ckpt["value"])
    tn.optimizer.load_state_dict(ckpt["transition_opt"])
    en.optimizer.load_state_dict(ckpt["terminal_opt"])
    vn.optimizer.load_state_dict(ckpt["value_opt"])

    if ckpt["target_net"] is not None:
        pt.target_net = deepcopy(vn.layers)
        pt.target_net.load_state_dict(ckpt["target_net"])

    d = ckpt["dataset"]
    ds.episodes = d["episodes"]
    ds.goal_episodes = d["goal_episodes"]
    ds.cum_rew = d["cum_rew"]
    ds.add_idx = d["add_idx"]
    ds.n_samples = d["n_samples"]
    ds.max_ep_len = d["max_ep_len"]
    ds.min_ep_len = d["min_ep_len"]
    ds.episode_add_idx = d["episode_add_idx"]
    ds.total_num_transitions = d["total_num_transitions"]
    ds.replace = d["replace"]
    ds.goal_add_idx = d["goal_add_idx"]
    ds.goal_n_samples = d["goal_n_samples"]
    ds.goal_replace = d["goal_replace"]
    ds.random_state.set_state(d["rng"])

    agent.random_state.set_state(ckpt["agent_rng"])
    torch.set_rng_state(ckpt["torch_rng"].to("cpu") if hasattr(ckpt["torch_rng"], "to") else ckpt["torch_rng"])
    if ckpt["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.to("cpu") for s in ckpt["cuda_rng"]])
    np.random.set_state(ckpt["numpy_rng"])

    return ckpt["experiment_step"], ckpt["ep"]
