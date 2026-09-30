"""
Chained hyperparameter search for PSDRL on DeepSea 5x5 (deterministic).

No autoencoder: DeepSea uses 25-dim one-hot vector observations.
  embed_dim = obs_dim = 25  (identity; raw obs are the embeddings)

Why the AE exists in PSDRL:
  The original paper targets Atari (64x64 grayscale images, 4096-dim raw obs).
  A CNN AE compresses each frame to embed_dim=1536 so the GRU and BLR
  operate in a tractable latent space. For DeepSea, observations are already
  25-dim binary vectors — no compression is needed or beneficial.

Two search stages:
  1. Transition Network + BLR  — gru_dim, hidden_dim, lr, window_length,
                                 reward_prior, transition_prior
  2. Value Network             — hidden_dim, lr, discount, training_iterations
                                 (input dim = embed_dim + gru_dim from stage 1)

Run from BRL/src/:
    python param_tuning/run_psdrl_search.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random, json, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
from optuna.samplers import NSGAIISampler
from pathlib import Path

from bsuite.environments import deep_sea
from PSDRL.networks.transition import Network as TransitionNetwork
from PSDRL.networks.value import Network as ValueNetwork
from PSDRL.common.settings import BLR_COEFFICIENT, ONE_OVER_LAMBDA

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

# GPU with safe fallback
def get_device():
    if not torch.cuda.is_available():
        return 'cpu'
    try:
        torch.zeros(1).cuda()
        torch.cuda.synchronize()
        return 'cuda'
    except RuntimeError:
        return 'cpu'

DEVICE = get_device()
print(f"Device: {DEVICE}")

# ─── Fixed constants ──────────────────────────────────────────────────────────
DS_SIZE     = 5
N_ACTIONS   = 2
EMBED_DIM   = DS_SIZE * DS_SIZE   # 25 — obs IS the embedding (no AE)
N_GRID      = DS_SIZE
NOISE_VAR   = ONE_OVER_LAMBDA     # 1.0 (from PSDRL settings)

TIME_LIMIT  = 1000
TEST_SIZE   = 200
PROBE_SIZES = [250, 500]
PATIENCE    = 15
N_TRIALS    = 40
BATCH_SIZE  = 16    # episodes per gradient step (fixed)
N_HYPOTHESES = 20   # BLR posterior samples for evaluation
SEED        = 42

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
Path('configs').mkdir(exist_ok=True)


# ─── Data collection ──────────────────────────────────────────────────────────
def collect_episodes(n_transitions, seed=SEED):
    env = deep_sea.DeepSea(size=DS_SIZE, deterministic=True, seed=seed)
    episodes, total = [], 0
    while total < n_transitions:
        ts = env.reset()
        obs_l, act_l, next_l, rew_l = [], [], [], []
        while True:
            obs    = torch.FloatTensor(ts.observation.flatten())
            action = np.random.randint(0, N_ACTIONS)
            ts     = env.step(action)
            obs_l.append(obs); act_l.append(action)
            next_l.append(torch.FloatTensor(ts.observation.flatten()))
            rew_l.append(float(ts.reward))
            total += 1
            if ts.last():
                break
        episodes.append({
            'obs':      torch.stack(obs_l),
            'actions':  torch.LongTensor(act_l),
            'next_obs': torch.stack(next_l),
            'rewards':  torch.FloatTensor(rew_l),
        })
    return episodes


def episodes_from_n(episodes, n):
    result, total = [], 0
    for ep in episodes:
        result.append(ep)
        total += len(ep['obs'])
        if total >= n:
            break
    return result


def param_count(model):
    return sum(p.numel() for p in model.parameters())


print("Collecting DeepSea 5x5 episodes...")
all_eps   = collect_episodes(TIME_LIMIT + TEST_SIZE)
cumlen    = np.cumsum([len(ep['obs']) for ep in all_eps])
train_cut = next(i for i, c in enumerate(cumlen) if c >= TIME_LIMIT)
train_eps = all_eps[:train_cut]
test_eps  = all_eps[train_cut:train_cut + max(1, TEST_SIZE // DS_SIZE)]
n_train   = sum(len(ep['obs']) for ep in train_eps)
n_test    = sum(len(ep['obs']) for ep in test_eps)
print(f"  Train: {len(train_eps)} eps / {n_train} transitions")
print(f"  Test:  {len(test_eps)} eps / {n_test} transitions")
print(f"  embed_dim = {EMBED_DIM} (raw obs, no autoencoder)\n")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1: Transition Network + Bayesian Linear Regression
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 62)
print("STEP 1 — GRU Transition Network + BLR")
print("  Tuning: gru_dim, hidden_dim, learning_rate, window_length,")
print("          reward_prior, transition_prior")
print("=" * 62)

TM_GRU_CHOICES    = [32, 64, 128, 256]
TM_HIDDEN_CHOICES = [64, 128, 256]
TM_LR_CHOICES     = [1e-4, 3e-4, 1e-3]
TM_WINDOW_CHOICES = [1, 2, 4]
PRIOR_CHOICES     = [1e-4, 1e-3, 1e-2, 0.1]   # reward_prior & transition_prior
TM_MAX_ITERS      = 200
TM_PROBE_ITERS    = 100
TARGET_GRID_DIST  = 0.5


# ── Transition network training (TBPTT) ──────────────────────────────────────

def build_ep_tensors(episodes, device):
    max_T = max(len(ep['obs']) for ep in episodes)
    N     = len(episodes)
    obs_t  = torch.zeros(N, max_T, EMBED_DIM)
    act_t  = torch.zeros(N, max_T, dtype=torch.long)
    next_t = torch.zeros(N, max_T, EMBED_DIM)
    rew_t  = torch.zeros(N, max_T, 1)
    lens   = [len(ep['obs']) for ep in episodes]
    for i, ep in enumerate(episodes):
        T = lens[i]
        obs_t[i, :T]  = ep['obs']
        act_t[i, :T]  = ep['actions']
        next_t[i, :T] = ep['next_obs']
        rew_t[i, :T]  = ep['rewards'].unsqueeze(-1)
    return (obs_t.to(device), act_t.to(device),
            next_t.to(device), rew_t.to(device), lens, max_T)


def train_transition(net, episodes, n_iters, window_len, lr,
                     device=DEVICE):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()
    obs_t, act_t, next_t, rew_t, lens, max_T = build_ep_tensors(episodes, device)
    N = len(episodes)
    best, pat, e_conv = float('inf'), 0, n_iters

    for it in range(n_iters):
        idx    = torch.randint(0, N, (BATCH_SIZE,))
        obs_b  = obs_t[idx]; act_b  = act_t[idx]
        next_b = next_t[idx]; rew_b = rew_t[idx]
        ep_lens = [lens[i.item()] for i in idx]

        h = torch.zeros(BATCH_SIZE, net.gru_dim, device=device)
        it_loss, n_win = 0.0, 0

        for start in range(0, max_T, window_len):
            end = min(start + window_len, max_T)
            optimizer.zero_grad()
            w_loss = torch.tensor(0.0, device=device)

            for t in range(start, end):
                valid = [j for j in range(BATCH_SIZE) if t < ep_lens[j]]
                if not valid:
                    break
                vi     = torch.tensor(valid, device=device)
                s      = obs_b[vi, t]; a = act_b[vi, t]
                s1     = next_b[vi, t]; r = rew_b[vi, t]
                act_oh = F.one_hot(a, num_classes=N_ACTIONS).float()
                sa     = torch.cat([s, act_oh], dim=-1)
                pred, h_new = net.forward(sa, h[valid])
                h[valid] = h_new
                w_loss = w_loss + loss_fn(pred[:, :-1], s1) + loss_fn(pred[:, -1:], r)

            if w_loss.item() == 0.0:
                continue
            (w_loss / (end - start)).backward()
            optimizer.step()
            h = h.detach()
            it_loss += w_loss.item() / (end - start)
            n_win   += 1

        avg = it_loss / max(1, n_win)
        if avg < best - 1e-5:
            best, pat, e_conv = avg, 0, it + 1
        else:
            pat += 1
            if pat >= PATIENCE:
                break
    return best, e_conv


# ── BLR: fit posterior and sample ────────────────────────────────────────────

def compute_feature_maps(net, episodes, device=DEVICE):
    """Compute GRU feature maps for every transition in episodes."""
    feats, targets = [], []
    net.eval()
    with torch.no_grad():
        for ep in episodes:
            T = len(ep['obs'])
            h = torch.zeros(1, net.gru_dim, device=device)
            for t in range(T):
                s   = ep['obs'][t].unsqueeze(0).to(device)
                a   = ep['actions'][t].long().item()
                s1  = ep['next_obs'][t].to(device)
                r   = ep['rewards'][t].to(device)
                act_oh = F.one_hot(torch.tensor([a], device=device),
                                   num_classes=N_ACTIONS).float()
                sa = torch.cat([s, act_oh], dim=-1)
                h  = net._cell(sa, h)
                feat = net.layers[:-1](torch.cat([h, sa], dim=-1))
                feats.append(feat.squeeze(0))
                targets.append(torch.cat([s1, r.unsqueeze(0)], dim=-1))
    net.train()
    return torch.stack(feats), torch.stack(targets)


def fit_blr(feats, targets, transition_prior, reward_prior,
            device=DEVICE):
    """Fit BLR posteriors; returns (mu, transition_cov, reward_cov)."""
    x, y         = feats, targets
    latent_dim   = x.shape[1]
    eye          = torch.eye(latent_dim, device=device)
    Phi_pre      = x.T.matmul(x) * NOISE_VAR
    coeff        = float(BLR_COEFFICIENT)

    while True:
        mat  = Phi_pre.double() + eye.double() * coeff
        try:
            chol = torch.linalg.cholesky(mat)
            Phi  = torch.cholesky_inverse(chol).float()
        except torch.linalg.LinAlgError:
            coeff *= 10; continue
        if Phi.isnan().any():
            coeff *= 10; continue
        break

    transition_cov = Phi * transition_prior
    reward_cov     = Phi * reward_prior
    mu = torch.zeros(EMBED_DIM + 1, latent_dim, device=device)
    for i in range(EMBED_DIM + 1):
        mu[i] = (NOISE_VAR * Phi).matmul(x.T.matmul(y[:, i]))

    return mu, transition_cov, reward_cov


def sample_w(mu, transition_cov, reward_cov, device=DEVICE):
    """Draw one weight matrix from the BLR posterior."""
    rs = torch.randn_like(mu)
    w  = torch.empty_like(mu)
    w[:-1] = mu[:-1] + (rs[:-1] @ transition_cov)
    w[-1]  = mu[-1]  + (rs[-1]  @ reward_cov)
    return w


def eval_transition_blr(net, mu, transition_cov, reward_cov,
                        episodes, device=DEVICE):
    """Mean grid distance averaged over N_HYPOTHESES BLR samples."""
    net.eval()
    all_dists = []
    with torch.no_grad():
        for ep in episodes:
            T = len(ep['obs'])
            h = torch.zeros(1, net.gru_dim, device=device)
            for t in range(T):
                s   = ep['obs'][t].unsqueeze(0).to(device)
                a   = ep['actions'][t].long().item()
                s1  = ep['next_obs'][t]
                act_oh = F.one_hot(torch.tensor([a], device=device),
                                   num_classes=N_ACTIONS).float()
                sa   = torch.cat([s, act_oh], dim=-1)
                h    = net._cell(sa, h)
                feat = net.layers[:-1](torch.cat([h, sa], dim=-1)).squeeze(0)

                true_idx = s1.argmax().item()
                tr, tc   = divmod(true_idx, N_GRID)

                # Average over sampled models (Thompson Sampling analogue)
                step_dists = []
                for _ in range(N_HYPOTHESES):
                    w         = sample_w(mu, transition_cov, reward_cov, device)
                    pred_state = (w @ feat)[:-1]
                    pred_idx  = pred_state.argmax().item()
                    pr, pc    = divmod(pred_idx, N_GRID)
                    step_dists.append(((pr-tr)**2 + (pc-tc)**2)**0.5)
                all_dists.append(float(np.mean(step_dists)))

    net.train()
    return float(np.mean(all_dists)) if all_dists else float('inf')


# ── Optuna objective (transition + BLR) ──────────────────────────────────────

def tm_objective(trial):
    gru_dim    = trial.suggest_categorical('gru_dim',          TM_GRU_CHOICES)
    hidden_dim = trial.suggest_categorical('hidden_dim',       TM_HIDDEN_CHOICES)
    lr         = trial.suggest_categorical('learning_rate',    TM_LR_CHOICES)
    window_len = trial.suggest_categorical('window_length',    TM_WINDOW_CHOICES)
    rew_prior  = trial.suggest_categorical('reward_prior',     PRIOR_CHOICES)
    tr_prior   = trial.suggest_categorical('transition_prior', PRIOR_CHOICES)

    cfg = {'gru_dim': gru_dim, 'hidden_dim': hidden_dim, 'learning_rate': lr}

    # --- Objective 1: train on full dataset, eval with BLR ---
    net = TransitionNetwork(EMBED_DIM, N_ACTIONS, cfg, DEVICE)
    _, e_conv = train_transition(net, train_eps, TM_MAX_ITERS, window_len, lr)
    feats, tgts = compute_feature_maps(net, train_eps)
    mu, t_cov, r_cov = fit_blr(feats, tgts, tr_prior, rew_prior)
    mgd = eval_transition_blr(net, mu, t_cov, r_cov, test_eps)
    trial.set_user_attr('epochs_to_converge', e_conv)

    # --- Objective 2: env_steps_to_accuracy ---
    env_steps = TIME_LIMIT
    for n in PROBE_SIZES:
        p_eps = episodes_from_n(train_eps, n)
        p_net = TransitionNetwork(EMBED_DIM, N_ACTIONS, cfg, DEVICE)
        train_transition(p_net, p_eps, TM_PROBE_ITERS, window_len, lr)
        p_feats, p_tgts = compute_feature_maps(p_net, p_eps)
        p_mu, p_tc, p_rc = fit_blr(p_feats, p_tgts, tr_prior, rew_prior)
        if eval_transition_blr(p_net, p_mu, p_tc, p_rc, test_eps) < TARGET_GRID_DIST:
            env_steps = n
            break

    return mgd, float(env_steps)


tm_study = optuna.create_study(
    directions=['minimize', 'minimize'],
    sampler=NSGAIISampler(seed=SEED),
)
tm_study.optimize(tm_objective, n_trials=N_TRIALS, show_progress_bar=True)

# ── Select best trial from Pareto front ──────────────────────────────────────
tm_rows = []
for t in tm_study.trials:
    if t.values is None:
        continue
    r = dict(t.params)
    r['mean_grid_dist'] = t.values[0]
    r['env_steps']      = t.values[1]
    r['epochs_to_converge'] = t.user_attrs.get('epochs_to_converge', TM_MAX_ITERS)
    r['pareto'] = t in tm_study.best_trials
    tm_rows.append(r)

tm_pareto = [r for r in tm_rows if r['pareto']]
candidates = tm_pareto if tm_pareto else tm_rows
mgd_vals  = [r['mean_grid_dist'] for r in candidates]
step_vals = [r['env_steps']      for r in candidates]
mgd_rng   = max(mgd_vals) - min(mgd_vals)
stp_rng   = max(step_vals) - min(step_vals)
best_tm_row = min(
    candidates,
    key=lambda r: (
        (r['mean_grid_dist'] - min(mgd_vals)) / (mgd_rng or 1)
        + (r['env_steps'] - min(step_vals)) / (stp_rng or 1)
    )
)

OPT_GRU_DIM   = int(best_tm_row['gru_dim'])
OPT_TM_HIDDEN = int(best_tm_row['hidden_dim'])
OPT_TM_LR     = float(best_tm_row['learning_rate'])
OPT_TM_WINDOW = int(best_tm_row['window_length'])
OPT_TR_PRIOR  = float(best_tm_row['transition_prior'])
OPT_RW_PRIOR  = float(best_tm_row['reward_prior'])

# ── Retrain best model and compute final metrics ──────────────────────────────
best_tm_cfg = {'gru_dim': OPT_GRU_DIM, 'hidden_dim': OPT_TM_HIDDEN,
               'learning_rate': OPT_TM_LR}
best_tm = TransitionNetwork(EMBED_DIM, N_ACTIONS, best_tm_cfg, DEVICE)
tm_final_loss, tm_epochs = train_transition(
    best_tm, train_eps, TM_MAX_ITERS, OPT_TM_WINDOW, OPT_TM_LR)
tm_feats, tm_tgts = compute_feature_maps(best_tm, train_eps)
tm_mu, tm_tcov, tm_rcov = fit_blr(tm_feats, tm_tgts, OPT_TR_PRIOR, OPT_RW_PRIOR)
tm_test_mgd = eval_transition_blr(best_tm, tm_mu, tm_tcov, tm_rcov, test_eps)
tm_nparams  = param_count(best_tm)

print(f"\nBest Transition+BLR:")
print(f"  gru_dim={OPT_GRU_DIM}, hidden_dim={OPT_TM_HIDDEN}, lr={OPT_TM_LR}, "
      f"window={OPT_TM_WINDOW}")
print(f"  transition_prior={OPT_TR_PRIOR}, reward_prior={OPT_RW_PRIOR}")
print(f"  Pareto: mean_grid_dist={best_tm_row['mean_grid_dist']:.4f}, "
      f"env_steps={best_tm_row['env_steps']:.0f}")
print(f"  Retrained → loss={tm_final_loss:.6f}, test_mgd={tm_test_mgd:.4f}")
print(f"  Transition net parameters: {tm_nparams:,}")

tm_params_out = {
    'gru_dim':               OPT_GRU_DIM,
    'hidden_dim':            OPT_TM_HIDDEN,
    'learning_rate':         OPT_TM_LR,
    'window_length':         OPT_TM_WINDOW,
    'transition_prior':      OPT_TR_PRIOR,
    'reward_prior':          OPT_RW_PRIOR,
    'embed_dim':             EMBED_DIM,
    'mean_grid_dist':        float(best_tm_row['mean_grid_dist']),
    'env_steps_to_accuracy': int(best_tm_row['env_steps']),
    'epochs_to_converge':    int(best_tm_row['epochs_to_converge']),
    'final_train_loss':      float(tm_final_loss),
    'final_test_mgd':        float(tm_test_mgd),
    'n_parameters':          tm_nparams,
    'n_hypotheses_eval':     N_HYPOTHESES,
    'n_actions': N_ACTIONS, 'ds_size': DS_SIZE,
    'n_trials': N_TRIALS, 'data_samples': TIME_LIMIT,
}
with open('configs/psdrl_transition_params.json', 'w') as f:
    json.dump(tm_params_out, f, indent=2)
print(f"  Saved → configs/psdrl_transition_params.json")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2: Value Network
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*62}")
print(f"STEP 2 — Value Network")
print(f"  embed_dim={EMBED_DIM}, gru_dim={OPT_GRU_DIM} "
      f"→ input_dim={EMBED_DIM + OPT_GRU_DIM}")
print(f"  Tuning: hidden_dim, learning_rate, discount, training_iterations")
print(f"{'='*62}")

VN_HIDDEN_CHOICES = [32, 64, 128, 256]
VN_LR_CHOICES     = [1e-4, 5e-4, 1e-3, 3e-3]
VN_DISC_CHOICES   = [0.9, 0.95, 0.99]
VN_ITER_CHOICES   = [50, 100, 200]
TARGET_ERR        = 0.05
VN_BATCH          = 64


def collect_value_inputs(episodes, tm, device=DEVICE):
    """Produce (embed_dim + gru_dim) inputs and raw reward lists."""
    inputs, rew_lists = [], []
    tm.eval()
    with torch.no_grad():
        for ep in episodes:
            T   = len(ep['obs'])
            h   = torch.zeros(1, tm.gru_dim, device=device)
            h_list = []
            for t in range(T):
                h_list.append(h.squeeze(0).cpu())
                s   = ep['obs'][t].unsqueeze(0).to(device)
                a   = ep['actions'][t].long().item()
                act_oh = F.one_hot(torch.tensor([a], device=device),
                                   num_classes=N_ACTIONS).float()
                sa  = torch.cat([s, act_oh], dim=-1)
                _, h = tm.forward(sa, h)
            for t in range(T):
                inp = torch.cat([ep['obs'][t], h_list[t]], dim=-1)
                inputs.append(inp)
            rew_lists.append(ep['rewards'].numpy())
    tm.train()
    return torch.stack(inputs), rew_lists


def mc_returns(rew_lists, discount):
    rets = []
    for rews in rew_lists:
        G = 0.0
        ep_rets = []
        for r in reversed(rews):
            G = r + discount * G
            ep_rets.insert(0, G)
        rets.extend(ep_rets)
    return torch.FloatTensor(rets).unsqueeze(-1)


print("  Pre-computing value inputs (raw obs + GRU hidden states)...")
train_vn_inp, train_rew = collect_value_inputs(train_eps, best_tm)
test_vn_inp,  test_rew  = collect_value_inputs(test_eps,  best_tm)
print(f"  Value input shape: {train_vn_inp.shape}")


def build_vnet(hidden_dim, lr, device=DEVICE):
    cfg = {'hidden_dim': hidden_dim, 'learning_rate': lr}
    return ValueNetwork(EMBED_DIM, cfg, device, gru_dim=OPT_GRU_DIM)


def train_vnet(vnet, inputs, returns, n_iters, device=DEVICE):
    N = len(inputs)
    inp_t = inputs.to(device); ret_t = returns.to(device)
    loss_fn = nn.MSELoss()
    best, pat, e_conv = float('inf'), 0, n_iters
    for i in range(n_iters):
        idx = torch.randint(0, N, (VN_BATCH,))
        vnet.optimizer.zero_grad()
        loss = loss_fn(vnet.forward(inp_t[idx]), ret_t[idx])
        loss.backward(); vnet.optimizer.step()
        if loss.item() < best - 1e-5:
            best, pat, e_conv = loss.item(), 0, i + 1
        else:
            pat += 1
            if pat >= PATIENCE: break
    return best, e_conv


def eval_vnet(vnet, inputs, returns, device=DEVICE):
    vnet.eval()
    with torch.no_grad():
        err = nn.MSELoss()(vnet.forward(inputs.to(device)),
                           returns.to(device)).item()
    vnet.train()
    return err


def vn_objective(trial):
    hidden_dim = trial.suggest_categorical('hidden_dim',           VN_HIDDEN_CHOICES)
    lr         = trial.suggest_categorical('learning_rate',        VN_LR_CHOICES)
    discount   = trial.suggest_categorical('discount',             VN_DISC_CHOICES)
    n_iters    = trial.suggest_categorical('training_iterations',  VN_ITER_CHOICES)

    train_ret = mc_returns(train_rew, discount)
    test_ret  = mc_returns(test_rew,  discount)

    vnet = build_vnet(hidden_dim, lr)
    _, e_conv = train_vnet(vnet, train_vn_inp, train_ret, n_iters)
    err = eval_vnet(vnet, test_vn_inp, test_ret)
    trial.set_user_attr('epochs_to_converge', e_conv)

    env_steps = TIME_LIMIT
    for n in PROBE_SIZES:
        p_inp, p_rew = collect_value_inputs(episodes_from_n(train_eps, n), best_tm)
        p_ret  = mc_returns(p_rew, discount)
        p_vnet = build_vnet(hidden_dim, lr)
        train_vnet(p_vnet, p_inp, p_ret, n_iters // 2)
        if eval_vnet(p_vnet, test_vn_inp, test_ret) < TARGET_ERR:
            env_steps = n
            break

    return err, float(env_steps)


vn_study = optuna.create_study(
    directions=['minimize', 'minimize'],
    sampler=NSGAIISampler(seed=SEED),
)
vn_study.optimize(vn_objective, n_trials=N_TRIALS, show_progress_bar=True)

vn_rows = []
for t in vn_study.trials:
    if t.values is None:
        continue
    r = dict(t.params)
    r['return_error'] = t.values[0]; r['env_steps'] = t.values[1]
    r['epochs_to_converge'] = t.user_attrs.get('epochs_to_converge', 200)
    r['pareto'] = t in vn_study.best_trials
    vn_rows.append(r)

vn_pareto = [r for r in vn_rows if r['pareto']]
cands = vn_pareto if vn_pareto else vn_rows
er_v  = [r['return_error'] for r in cands]; st_v = [r['env_steps'] for r in cands]
er_r  = max(er_v) - min(er_v); sr = max(st_v) - min(st_v)
best_vn_row = min(cands,
    key=lambda r: (r['return_error'] - min(er_v)) / (er_r or 1)
                  + (r['env_steps'] - min(st_v)) / (sr or 1))

OPT_VN_HIDDEN = int(best_vn_row['hidden_dim'])
OPT_VN_LR     = float(best_vn_row['learning_rate'])
OPT_VN_DISC   = float(best_vn_row['discount'])
OPT_VN_ITERS  = int(best_vn_row['training_iterations'])

best_train_ret = mc_returns(train_rew, OPT_VN_DISC)
best_test_ret  = mc_returns(test_rew,  OPT_VN_DISC)
best_vn = build_vnet(OPT_VN_HIDDEN, OPT_VN_LR)
vn_final_loss, vn_epochs = train_vnet(best_vn, train_vn_inp, best_train_ret, OPT_VN_ITERS)
vn_test_err  = eval_vnet(best_vn, test_vn_inp, best_test_ret)
vn_nparams   = param_count(best_vn)

print(f"\nBest Value Network:")
print(f"  hidden_dim={OPT_VN_HIDDEN}, lr={OPT_VN_LR}, "
      f"discount={OPT_VN_DISC}, iters={OPT_VN_ITERS}")
print(f"  Pareto: return_error={best_vn_row['return_error']:.5f}, "
      f"env_steps={best_vn_row['env_steps']:.0f}")
print(f"  Retrained → loss={vn_final_loss:.6f}, test_error={vn_test_err:.6f}")
print(f"  Value net parameters: {vn_nparams:,}")

vn_input_dim = EMBED_DIM + OPT_GRU_DIM
vn_params_out = {
    'hidden_dim':            OPT_VN_HIDDEN,
    'learning_rate':         OPT_VN_LR,
    'discount':              OPT_VN_DISC,
    'training_iterations':   OPT_VN_ITERS,
    'return_error':          float(best_vn_row['return_error']),
    'env_steps_to_accuracy': int(best_vn_row['env_steps']),
    'epochs_to_converge':    int(best_vn_row['epochs_to_converge']),
    'final_train_loss':      float(vn_final_loss),
    'final_test_error':      float(vn_test_err),
    'n_parameters':          vn_nparams,
    'embed_dim':             EMBED_DIM,
    'gru_dim':               OPT_GRU_DIM,
    'input_dim':             vn_input_dim,
    'n_actions': N_ACTIONS, 'ds_size': DS_SIZE,
    'n_trials': N_TRIALS, 'data_samples': TIME_LIMIT,
}
with open('configs/psdrl_value_params.json', 'w') as f:
    json.dump(vn_params_out, f, indent=2)
print(f"  Saved → configs/psdrl_value_params.json")


# ═════════════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*62}")
print("SUMMARY — PSDRL Optimal Parameters for DeepSea 5×5")
print(f"{'='*62}")
print(f"\n  No autoencoder — embed_dim = {EMBED_DIM} (raw 25-dim one-hot obs)")
print(f"\n{'Network':<26} {'#Params':>10}   Key hyperparameters")
print("-" * 62)
print(f"{'GRU Transition':<26} {tm_nparams:>10,}   "
      f"gru={OPT_GRU_DIM}, hidden={OPT_TM_HIDDEN}, lr={OPT_TM_LR}, "
      f"window={OPT_TM_WINDOW}")
print(f"{'BLR (no params)':<26} {'—':>10}   "
      f"tr_prior={OPT_TR_PRIOR}, rw_prior={OPT_RW_PRIOR}")
print(f"{'Value Network':<26} {vn_nparams:>10,}   "
      f"hidden={OPT_VN_HIDDEN}, lr={OPT_VN_LR}, γ={OPT_VN_DISC}")
print(f"{'TOTAL (neural nets)':<26} {tm_nparams+vn_nparams:>10,}")
print(f"\n  Value input: {EMBED_DIM} (embed) + {OPT_GRU_DIM} (gru) = {vn_input_dim}-dim")
print(f"\nAll configs saved to BRL/src/configs/")
