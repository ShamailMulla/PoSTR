"""
Generates optuna_transition_hparam_search.ipynb programmatically.
Run once: python generate_optuna_notebook.py
"""
import nbformat

def md(src):  return nbformat.v4.new_markdown_cell(src)
def code(src): return nbformat.v4.new_code_cell(src)

# ── Cell sources ─────────────────────────────────────────────────────────────

TITLE = """\
# BTRL Transition Network — Hyperparameter Optimisation (Optuna)

## Multi-Objective Search: Convergence Speed vs Grid Prediction Error

**Environment**: DeepSea-5, deterministic (5×5 grid, 25 one-hot states)
**Config base**: `configs/config_vector-setting1.yaml`

**Objectives** (both minimised — Pareto-front via NSGA-II):
1. **Mean grid distance** — mean Euclidean distance in (row, col) space between
   each sampled hypothesis prediction and the true next state, averaged across
   100 world hypotheses M̂ and all test cases (lower = more accurate posterior)
2. **Epochs to convergence** — epoch at which early stopping triggers (lower = faster)

**Why grid distance, not flat-index accuracy or variance?**
DeepSea states are flat indices into an N×N grid: state `s` → `(s//N, s%N)`.
Flat-index arithmetic is misleading at row boundaries — indices 4 and 5 have
flat-diff=1 but are actually on opposite ends of adjacent rows (Euclidean dist≈4.1
in a 5×5 grid). Grid distance correctly treats states as spatial positions.

**Key design**: Evaluation always uses `lock_all_dropouts()` (EpisodeLockedDropout fix).
Each of the 100 iterations samples a fresh world hypothesis M̂; distance is measured
from each hypothesis prediction to the true state — this is the expected prediction
error under the posterior.

**Parameters tuned**: `hidden_dim`, `heads` (jointly), `learning_rate`,
`num_encoder_layers`, `forward_expansion`
"""

IMPORTS = """\
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from copy import deepcopy
from ruamel.yaml import YAML
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
pd.options.display.max_rows = 200
pd.options.display.float_format = '{:.5f}'.format

print(f'PyTorch {torch.__version__}  |  Optuna {optuna.__version__}')
"""

TRL_IMPORTS = """\
from TRL.networks.transition import Network as TransitionNetwork
from notebook_utils import (
    generate_transition_data,
    test_model_locked,
    test_model_unlocked,
    early_stopping_train,
)
"""

CONFIG_LOAD = """\
CONFIG_PATH = "configs/config_vector-setting1.yaml"
with open(CONFIG_PATH, "r") as f:
    yaml = YAML(typ='rt')
    config = yaml.load(f)

# Fixed overrides for this experiment
config['gpu']                           = False   # CPU only
config['experiment']['deterministic']   = True    # deterministic DeepSea
config['experiment']['env']             = '5'     # depth = 5

DROPOUT = 0.2  # fixed across all trials (same as test notebooks)

print('Config:', CONFIG_PATH)
print('Transition params:', dict(config['transition']))
print('Env:', config['experiment']['env'],
      '| Deterministic:', config['experiment']['deterministic'])
"""

DATA_GEN = """\
TIME_LIMIT = 1000   # transitions — generated ONCE and shared across all trials

episodes, test_transitions, obs_space, possible_actions = generate_transition_data(
    config, TIME_LIMIT
)

test_size = len(test_transitions) * len(possible_actions)
print(f'obs_space            : {obs_space}')
print(f'possible_actions     : {possible_actions.flatten().tolist()}')
print(f'test states          : {list(test_transitions.keys())}')
print(f'test_size (rows)     : {test_size}')

# Convert to tensors (shared across all Optuna trials)
states      = torch.tensor(episodes['states'],    dtype=torch.float32)
next_states = torch.tensor(episodes['next_states'], dtype=torch.float32)
timesteps   = torch.tensor(episodes['timestep'],  dtype=int)
rewards     = torch.tensor(episodes['rewards']).unsqueeze(1)
actions     = torch.tensor(episodes['actions'],   dtype=int).unsqueeze(1)

print(f'\\nstates shape    : {states.shape}')
print(f'timesteps shape : {timesteps.shape}')
"""

SEARCH_SPACE_MD = """\
## Hyperparameter Search Space

| Parameter | Optuna type | Range / choices | Base config |
|---|---|---|---|
| `hidden_dim` + `heads` | Joint int index (20 valid pairs) | `hidden_dim` ∈ {32,64,128,256,512}, `heads` ∈ {1,2,4,8}, `hidden_dim % heads == 0` | 32, 2 |
| `learning_rate` | Log-uniform float | [1e-4, 1e-2] | 1e-2 |
| `num_encoder_layers` | Integer | [1, 4] | 2 |
| `forward_expansion` | Categorical | {2, 4, 8} | 4 |

`hidden_dim` and `heads` are sampled as a single index into a pre-computed list of
20 valid pairs. This guarantees `head_dim = hidden_dim // heads` is always a valid
integer without requiring conditional sampling or trial pruning.

**Fixed params**: `context_length=2`, `dropout=0.2`, `env=5` (deterministic DeepSea-5)

## Early stopping
- Max epochs: 300
- Patience: 15 epochs without improvement
- Min delta: 5×10⁻⁴
"""

CONSTANTS = """\
# Valid (hidden_dim, heads) pairs: all combos where hidden_dim % heads == 0
VALID_PAIRS = [
    (h, n)
    for h in [32, 64, 128, 256, 512]
    for n in [1, 2, 4, 8]
    if h % n == 0
]  # 20 pairs

MAX_EPOCHS   = 300
PATIENCE     = 15
MIN_DELTA    = 5e-4
N_HYPOTHESES = 100   # M̂ samples per test case for variance measurement
N_TRIALS     = 100

print(f'{len(VALID_PAIRS)} valid (hidden_dim, heads) pairs:')
for p in VALID_PAIRS:
    print(' ', p)
"""

OBJECTIVE = """\
def objective(trial):
    # ── Hyperparameter sampling ──────────────────────────────────────────────
    pair_idx           = trial.suggest_int('hd_heads_idx', 0, len(VALID_PAIRS) - 1)
    hidden_dim, heads  = VALID_PAIRS[pair_idx]
    learning_rate      = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    num_encoder_layers = trial.suggest_int('num_encoder_layers', 1, 4)
    forward_expansion  = trial.suggest_categorical('forward_expansion', [2, 4, 8])

    # ── Log all params (hyperparams + fixed config) ──────────────────────────
    trial.set_user_attr('hidden_dim',          hidden_dim)
    trial.set_user_attr('heads',               heads)
    trial.set_user_attr('learning_rate',       learning_rate)
    trial.set_user_attr('num_encoder_layers',  num_encoder_layers)
    trial.set_user_attr('forward_expansion',   forward_expansion)
    trial.set_user_attr('context_length',      config['transition']['context_length'])
    trial.set_user_attr('dropout',             DROPOUT)
    trial.set_user_attr('env',                 config['experiment']['env'])
    trial.set_user_attr('deterministic',       config['experiment']['deterministic'])
    trial.set_user_attr('data_samples',        TIME_LIMIT)
    trial.set_user_attr('max_epochs',          MAX_EPOCHS)
    trial.set_user_attr('patience',            PATIENCE)
    trial.set_user_attr('min_delta',           MIN_DELTA)

    # ── Build model ──────────────────────────────────────────────────────────
    trial_cfg = dict(config['transition'])
    trial_cfg['hidden_dim']         = hidden_dim
    trial_cfg['heads']              = heads
    trial_cfg['learning_rate']      = learning_rate
    trial_cfg['num_encoder_layers'] = num_encoder_layers
    trial_cfg['forward_expansion']  = forward_expansion

    model = TransitionNetwork(
        trial_cfg, obs_space, possible_actions,
        dropout=DROPOUT,
        max_steps=int(torch.max(timesteps).item()) + 1,
        device=torch.device('cpu'),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ── Train with early stopping ────────────────────────────────────────────
    loss_history, epochs_to_converge, best_loss = early_stopping_train(
        model, states, next_states, timesteps, actions, rewards, optimizer,
        max_epochs=MAX_EPOCHS, patience=PATIENCE, min_delta=MIN_DELTA,
    )

    # ── Evaluate: grid-space metrics across N_HYPOTHESES sampled M̂ ─────────────
    # Each iteration: lock fresh M̂ → predict all test cases → unlock.
    # mean_grid_dist = expected Euclidean distance in (row,col) space from
    #                  predicted to true next state, averaged over all M̂ and cases.
    # grid_variance  = var(pred_rows) + var(pred_cols) across M̂ per case (mean).
    results = test_model_locked(
        model, test_transitions, possible_actions, n_hypotheses=N_HYPOTHESES
    )

    correct         = int(results['exact_match'].sum())
    mean_grid_dist  = float(results['mean_grid_dist'].mean())
    max_grid_dist   = float(results['mean_grid_dist'].max())
    mean_grid_var   = float(results['grid_variance'].mean())

    trial.set_user_attr('correct_preds',      correct)
    trial.set_user_attr('test_size',          test_size)
    trial.set_user_attr('mean_grid_dist',     mean_grid_dist)
    trial.set_user_attr('max_grid_dist',      max_grid_dist)
    trial.set_user_attr('mean_grid_variance', mean_grid_var)
    trial.set_user_attr('epochs_to_converge', epochs_to_converge)
    trial.set_user_attr('best_loss',          best_loss)

    return mean_grid_dist, float(epochs_to_converge)
"""

RUN_STUDY = """\
sampler = optuna.samplers.NSGAIISampler(seed=42)
study = optuna.create_study(
    directions=['minimize', 'minimize'],
    sampler=sampler,
    study_name='btrl_transition_hparam_search',
)

print(f'Running {N_TRIALS} trials')
print('Objectives: [0] mean_grid_dist (Euclidean, grid space)  [1] epochs_to_converge')
print('Both minimised via NSGA-II Pareto-front search')
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
print(f'\\nDone.  Pareto-front trials: {len(study.best_trials)}')
"""

EXTRACT_RESULTS = """\
def trials_to_df(study):
    records = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        ua = t.user_attrs
        records.append({
            'trial':              t.number,
            'hidden_dim':         ua.get('hidden_dim'),
            'heads':              ua.get('heads'),
            'learning_rate':      ua.get('learning_rate'),
            'num_encoder_layers': ua.get('num_encoder_layers'),
            'forward_expansion':  ua.get('forward_expansion'),
            'context_length':     ua.get('context_length'),
            'dropout':            ua.get('dropout'),
            'mean_grid_dist':     ua.get('mean_grid_dist'),     # objective 1 (grid space)
            'max_grid_dist':      ua.get('max_grid_dist'),
            'mean_grid_variance': ua.get('mean_grid_variance'), # posterior spread, grid space
            'epochs_to_converge': ua.get('epochs_to_converge'), # objective 2
            'correct_preds':      ua.get('correct_preds'),
            'test_size':          ua.get('test_size'),
            'best_loss':          ua.get('best_loss'),
        })
    df = pd.DataFrame(records)
    df['accuracy'] = df['correct_preds'] / df['test_size']
    return df.sort_values('mean_grid_dist').reset_index(drop=True)

results_df = trials_to_df(study)
print(f'{len(results_df)} completed trials')
results_df
"""

PARETO_PLOT = """\
pareto_df = pd.DataFrame([{
    'mean_grid_dist':     t.values[0],
    'epochs_to_converge': t.values[1],
    'hidden_dim':         t.user_attrs.get('hidden_dim'),
    'accuracy':           t.user_attrs.get('correct_preds', 0) /
                          max(t.user_attrs.get('test_size', 1), 1),
} for t in study.best_trials])

fig, ax = plt.subplots(figsize=(11, 6))
sc = ax.scatter(
    results_df['epochs_to_converge'], results_df['mean_grid_dist'],
    c=results_df['accuracy'], cmap='RdYlGn', alpha=0.65, s=70,
    vmin=0, vmax=1, edgecolors='grey', linewidths=0.3,
)
ax.scatter(
    pareto_df['epochs_to_converge'], pareto_df['mean_grid_dist'],
    color='black', marker='*', s=250,
    label=f'Pareto front ({len(pareto_df)} trials)', zorder=5,
)
plt.colorbar(sc, ax=ax, label='Exact-match accuracy (correct / test_size)')
ax.set_xlabel('Epochs to convergence (early stopping)')
ax.set_ylabel('Mean grid distance (Euclidean, row-col space)')
ax.set_title(
    'Pareto Front — BTRL Transition Hyperparameter Search\\n'
    'DeepSea-5 Deterministic  |  Objectives: grid_dist ↓, epochs ↓'
)
ax.legend()
plt.tight_layout()
plt.show()

print('\\nPareto-front configs:')
print(pareto_df.sort_values('mean_grid_dist').to_string(index=False))
"""

SCATTER_PLOTS = """\
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: epochs vs grid distance, coloured by hidden_dim
dims   = sorted(results_df['hidden_dim'].dropna().unique())
cmap_d = plt.cm.get_cmap('tab10', len(dims))
for i, d in enumerate(dims):
    sub = results_df[results_df['hidden_dim'] == d]
    axes[0].scatter(
        sub['epochs_to_converge'], sub['mean_grid_dist'],
        label=f'dim={int(d)}', color=cmap_d(i), alpha=0.7, s=55,
    )
axes[0].set_xlabel('Epochs to convergence')
axes[0].set_ylabel('Mean grid distance (Euclidean)')
axes[0].set_title('Convergence vs Grid Distance  (coloured by hidden_dim)')
axes[0].legend(fontsize=8)

# Right: grid distance distribution by num_encoder_layers
results_df.boxplot(column='mean_grid_dist', by='num_encoder_layers', ax=axes[1])
axes[1].set_title('Grid Distance by num_encoder_layers')
axes[1].set_xlabel('num_encoder_layers')
axes[1].set_ylabel('Mean grid distance')
plt.suptitle('')

plt.tight_layout()
plt.show()
"""

TOP5_TABLES = """\
cols = ['trial', 'hidden_dim', 'heads', 'learning_rate', 'num_encoder_layers',
        'forward_expansion', 'mean_grid_dist', 'mean_grid_variance',
        'epochs_to_converge', 'accuracy', 'best_loss']

print('Top 5 — lowest mean grid distance (most spatially accurate posterior):')
display(results_df.nsmallest(5, 'mean_grid_dist')[cols])

print('\\nTop 5 — fastest convergence (fewest epochs):')
display(results_df.nsmallest(5, 'epochs_to_converge')[cols])

print('\\nTop 5 — highest exact-match accuracy:')
display(results_df.nlargest(5, 'accuracy')[cols])

# Best balanced: minimise normalised grid_dist + epochs + (1 - accuracy)
r = results_df.copy()
def norm(col): return (col - col.min()) / (col.max() - col.min() + 1e-9)
r['score'] = norm(r['mean_grid_dist']) + norm(r['epochs_to_converge']) + (1 - r['accuracy'])
best_idx   = r['score'].idxmin()
best_row   = r.loc[best_idx]

print('\\nBest balanced trial (min normalised grid_dist + epochs + accuracy_loss):')
display(r.loc[[best_idx], cols + ['score']])
"""

RETRAIN_BEST = """\
# Re-identify best balanced row (recompute in case cell is re-run independently)
r2 = results_df.copy()
def norm2(col): return (col - col.min()) / (col.max() - col.min() + 1e-9)
r2['score'] = norm2(r2['mean_grid_dist']) + norm2(r2['epochs_to_converge']) + (1 - r2['accuracy'])
best_row = r2.loc[r2['score'].idxmin()]

print('Retraining best balanced model:')
for k in ['hidden_dim', 'heads', 'learning_rate', 'num_encoder_layers',
          'forward_expansion', 'mean_grid_dist', 'epochs_to_converge', 'accuracy']:
    print(f'  {k:22s}: {best_row[k]}')

best_cfg = dict(config['transition'])
best_cfg['hidden_dim']         = int(best_row['hidden_dim'])
best_cfg['heads']              = int(best_row['heads'])
best_cfg['learning_rate']      = float(best_row['learning_rate'])
best_cfg['num_encoder_layers'] = int(best_row['num_encoder_layers'])
best_cfg['forward_expansion']  = int(best_row['forward_expansion'])

best_model = TransitionNetwork(
    best_cfg, obs_space, possible_actions,
    dropout=DROPOUT,
    max_steps=int(torch.max(timesteps).item()) + 1,
    device=torch.device('cpu'),
)
best_opt = torch.optim.Adam(best_model.parameters(), lr=best_cfg['learning_rate'])

loss_hist, conv_ep, final_loss = early_stopping_train(
    best_model, states, next_states, timesteps, actions, rewards, best_opt,
    max_epochs=MAX_EPOCHS, patience=PATIENCE, min_delta=MIN_DELTA,
)

plt.figure(figsize=(10, 4))
plt.plot(loss_hist, label='train loss', color='steelblue')
if conv_ep < MAX_EPOCHS:
    plt.axvline(conv_ep - PATIENCE, color='red', linestyle='--',
                label=f'early stop (epoch {conv_ep})')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
dim_h = int(best_row['hidden_dim'])
lr_v  = best_row['learning_rate']
plt.title(f'Best balanced model — hidden_dim={dim_h}, lr={lr_v:.2e}, '
          f'layers={int(best_row["num_encoder_layers"])}')
plt.legend()
plt.tight_layout()
plt.show()

print(f'Converged at epoch : {conv_ep}  |  Best loss : {final_loss:.6f}')
"""

LOCKED_DEMO = """\
# ── Locked vs Unlocked: grid distance and grid variance ──────────────────────
# Locked:   N_HYPOTHESES different M̂ sampled; metrics in grid (row,col) space
# Unlocked: fresh random mask every predict() call (pre-fix Dithering Trap)

res_locked   = test_model_locked(best_model, test_transitions, possible_actions,
                                  n_hypotheses=N_HYPOTHESES)
res_unlocked = test_model_unlocked(best_model, test_transitions, possible_actions,
                                    n_calls=N_HYPOTHESES)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Row 1: mean grid distance
axes[0, 0].bar(range(test_size), res_unlocked['mean_grid_dist'], color='tomato')
axes[0, 0].set_title('Grid Distance — UNLOCKED (Dithering Trap)\\nfresh mask every call')
axes[0, 0].set_xlabel('Test case'); axes[0, 0].set_ylabel('Mean grid dist (cells)')

axes[0, 1].bar(range(test_size), res_locked['mean_grid_dist'], color='steelblue')
axes[0, 1].set_title('Grid Distance — LOCKED (Thompson Sampling)\\n100 M̂ via lock_all_dropouts()')
axes[0, 1].set_xlabel('Test case'); axes[0, 1].set_ylabel('Mean grid dist (cells)')

# Row 2: grid variance (posterior spread in row-col space)
axes[1, 0].bar(range(test_size), res_unlocked['grid_variance'], color='tomato', alpha=0.8)
axes[1, 0].set_title('Grid Variance — UNLOCKED\\nvar(pred_rows) + var(pred_cols)')
axes[1, 0].set_xlabel('Test case'); axes[1, 0].set_ylabel('Grid variance')

axes[1, 1].bar(range(test_size), res_locked['grid_variance'], color='steelblue', alpha=0.8)
axes[1, 1].set_title('Grid Variance — LOCKED\\ninter-hypothesis spread in grid space')
axes[1, 1].set_xlabel('Test case'); axes[1, 1].set_ylabel('Grid variance')

dim_h = int(best_row['hidden_dim'])
lr_v  = best_row['learning_rate']
plt.suptitle(f'Best Optuna model  |  dim={dim_h}, heads={int(best_row["heads"])}, '
             f'lr={lr_v:.2e}', fontsize=12)
plt.tight_layout()
plt.show()

n_unlocked = int(res_unlocked['exact_match'].sum())
n_locked   = int(res_locked['exact_match'].sum())
print(f'Unlocked : correct={n_unlocked}/{test_size}, '
      f'mean_grid_dist={res_unlocked["mean_grid_dist"].mean():.4f}, '
      f'grid_var={res_unlocked["grid_variance"].mean():.4f}')
print(f'Locked   : correct={n_locked}/{test_size},   '
      f'mean_grid_dist={res_locked["mean_grid_dist"].mean():.4f}, '
      f'grid_var={res_locked["grid_variance"].mean():.4f}')
print()
print('Comparison table:')
comparison = pd.DataFrame({
    'state':              res_unlocked['current_state'].values,
    'action':             res_unlocked['action'].values,
    'true':               res_unlocked['true next_state'].values,
    'true(row,col)':      list(zip(res_unlocked['true_row'], res_unlocked['true_col'])),
    'pred(unlocked)':     res_unlocked['pred next_state'].values,
    'grid_dist(unlocked)':res_unlocked['mean_grid_dist'].values,
    'grid_var(unlocked)': res_unlocked['grid_variance'].values,
    'pred(locked)':       res_locked['pred next_state'].values,
    'grid_dist(locked)':  res_locked['mean_grid_dist'].values,
    'grid_var(locked)':   res_locked['grid_variance'].values,
})
display(comparison)
"""

# ── Assemble notebook ─────────────────────────────────────────────────────────

cells = [
    md(TITLE),
    code(IMPORTS),
    code(TRL_IMPORTS),
    code(CONFIG_LOAD),
    code(DATA_GEN),
    md(SEARCH_SPACE_MD),
    code(CONSTANTS),
    code(OBJECTIVE),
    code(RUN_STUDY),
    code(EXTRACT_RESULTS),
    code(PARETO_PLOT),
    code(SCATTER_PLOTS),
    code(TOP5_TABLES),
    code(RETRAIN_BEST),
    code(LOCKED_DEMO),
]

nb = nbformat.v4.new_notebook()
nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3',
    },
    'language_info': {
        'name': 'python',
        'version': '3.10.0',
    },
}

OUT = '/media/shamail/CRUCIAL/MS_QMUL/Thesis/BTRL/src/optuna_transition_hparam_search.ipynb'
nbformat.write(nb, OUT)
print(f'Written: {OUT}')
print(f'Cells: {len(nb["cells"])}')
