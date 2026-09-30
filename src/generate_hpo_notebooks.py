"""
Generate the three-stage HPO notebooks (hpo1..hpo3). Run with any python that
has nbformat installed; the notebooks themselves target the torch2_rl kernel.

    python3 generate_hpo_notebooks.py
"""
import nbformat as nbf

KERNEL = {
    "kernelspec": {"display_name": "torch2_rl", "language": "python", "name": "torch2_rl"},
    "language_info": {"name": "python", "version": "3.10.12"},
}


def build(path, cells):
    nb = nbf.v4.new_notebook()
    nb.metadata.update(KERNEL)
    nb.cells = [
        nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
        for kind, src in cells
    ]
    nbf.write(nb, path)
    print("wrote", path)


# ═══════════════════════════════ Notebook 1 ═══════════════════════════════

NB1 = [
("md", """\
# HPO Stage 1 — Transition Network (offline, multi-objective)

Tunes the BayesFormer transition network on a **fixed offline dataset** — no
environment interaction per trial, so trials are cheap and results are
attributable to the transition hyperparameters alone.

## Plan
1. **Setup** — load base config (`config_vector-setting1.yaml`, DeepSea-5
   deterministic), fix all seeds.
2. **Data** — collect one shared dataset of random-policy transitions
   (`generate_transition_data`) plus a per-state test set. Generated once,
   reused by every trial so scores are comparable.
3. **Objective** — per trial: sample hyperparameters, train with early
   stopping, evaluate with 100 locked-dropout world hypotheses M̂:
   - **Objective 1 ↓ `mean_grid_dist`** — mean Euclidean distance in
     (row, col) space between predicted and true next state. World-model
     accuracy.
   - **Objective 2 ↓ `overconfident_error_rate`** — fraction of
     (state, action) pairs that are *wrong* (grid dist > 0.5 cells) with a
     *near-zero posterior spread* (grid variance < 0.05). This is the failure
     mode that kills Thompson Sampling: if every sampled M̂ agrees on the wrong
     transition, no episode ever explores the correction. Accuracy alone
     cannot see it.
4. **Search** — NSGA-II Pareto-front search, persisted to the shared SQLite DB
   (`hpo_artifacts/optuna_btrl.db`) so the study is resumable.
5. **Analysis** — trial table + Pareto-front plot.
6. **Export** — Pareto candidates → `hpo_artifacts/transition_pareto.json`
   for the validation notebook (hpo2).

Diagnostics logged per trial (not searched on): reward-head MSE, goal-path
grid distance (bottom-half states), exact-match accuracy, parameter count.
"""),
("code", """\
import warnings
warnings.filterwarnings('ignore')

import json
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import optuna
from matplotlib import pyplot as plt

from notebook_utils import generate_transition_data
from TRL.tuning.common import (
    ARTIFACTS_DIR, OPTUNA_DB, TRANSITION_STUDY_NAME, TRANSITION_PARETO_JSON,
    load_base_config, set_global_seed, get_device,
)
from TRL.tuning import transition_search as ts

optuna.logging.set_verbosity(optuna.logging.WARNING)
pd.options.display.float_format = '{:.5f}'.format

config = load_base_config(env='5', deterministic=True)
DEVICE = get_device(config)
print(f'torch {torch.__version__} | optuna {optuna.__version__} | device {DEVICE}')"""),
("code", """\
# ── Budget knobs ─────────────────────────────────────────────────────────────
SEED         = 42
DATA_STEPS   = 2000   # offline dataset size (transitions), shared by all trials
N_TRIALS     = 200    # NSGA-II population needs breadth; trials are cheap
N_HYPOTHESES = 100    # locked-dropout M-hat samples per evaluation
MAX_EPOCHS   = 300
PATIENCE     = 15
MIN_DELTA    = 5e-4

set_global_seed(SEED)"""),
("code", """\
# ── Shared offline dataset (generated once, reused by every trial) ──────────
# Random policy on deterministic DeepSea-5. ~3% of random episodes reach the
# goal, so at 2000 transitions the goal region appears in both train and test.
episodes, test_transitions, obs_space, possible_actions = generate_transition_data(
    config, DATA_STEPS
)

data = {
    'states':      torch.tensor(episodes['states'],      dtype=torch.float32),
    'next_states': torch.tensor(episodes['next_states'], dtype=torch.float32),
    'timesteps':   torch.tensor(episodes['timestep'],    dtype=int),
    'rewards':     torch.tensor(episodes['rewards']).unsqueeze(1),
    'actions':     torch.tensor(episodes['actions'],     dtype=int).unsqueeze(1),
    'test_transitions': test_transitions,
    'obs_space': obs_space,
    'possible_actions': possible_actions,
}
test_size = len(test_transitions) * len(possible_actions)
print(f'obs_space={obs_space} | train transitions={len(episodes["actions"])} '
      f'| test (state,action) pairs={test_size}')"""),
("md", """\
## Search space

| Parameter | Range / choices | Represents |
|---|---|---|
| `hidden_dim` × `heads` | joint index over 20 valid pairs (dim ∈ {32…512}, heads ∈ {1,2,4,8}, dim % heads == 0) | embedding width and attention-head count — model capacity |
| `learning_rate` | log-uniform [1e-4, 1e-2] | Adam step size |
| `num_encoder_layers` | int [1, 4] | Transformer depth |
| `forward_expansion` | {2, 4, 8} | FFN width multiplier inside each block |
| `p_seq` | {0.05, 0.1, 0.2, 0.3} | embedding/positional dropout — observation-noise part of the Bayesian posterior |
| `p_attn` | {0.02, 0.05, 0.1, 0.15} | Q/K/V, attention-map and FFN dropout — epistemic-uncertainty part of the posterior |

`p_seq`/`p_attn` are not just regularisers here: the dropout masks **are** the
posterior over world models (BayesFormer), so these two directly shape the
Thompson-Sampling exploration distribution."""),
("code", """\
def objective(trial):
    params = ts.suggest_transition_params(trial)
    metrics = ts.train_and_eval_transition(
        config, params, data, DEVICE,
        max_epochs=MAX_EPOCHS, patience=PATIENCE, min_delta=MIN_DELTA,
        n_hypotheses=N_HYPOTHESES,
    )
    for key, value in {**params, **{k: v for k, v in metrics.items() if k != 'objectives'}}.items():
        trial.set_user_attr(key, value)
    return metrics['objectives']"""),
("code", """\
ARTIFACTS_DIR.mkdir(exist_ok=True)
study = optuna.create_study(
    study_name=TRANSITION_STUDY_NAME,
    storage=OPTUNA_DB,
    directions=['minimize', 'minimize'],
    sampler=optuna.samplers.NSGAIISampler(seed=SEED),
    load_if_exists=True,
)

done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
remaining = max(0, N_TRIALS - done)
print(f'{done} trials already in {OPTUNA_DB}; running {remaining} more')
if remaining:
    study.optimize(objective, n_trials=remaining, show_progress_bar=True)
print(f'Pareto-front size: {len(study.best_trials)}')"""),
("code", """\
# ── Trial table ──────────────────────────────────────────────────────────────
ATTRS = ['hidden_dim', 'heads', 'learning_rate', 'num_encoder_layers',
         'forward_expansion', 'p_seq', 'p_attn',
         'mean_grid_dist', 'overconfident_error_rate', 'reward_mse',
         'goal_path_grid_dist', 'exact_match_accuracy', 'mean_grid_variance',
         'epochs_to_converge', 'n_params']

results_df = pd.DataFrame([
    {'trial': t.number, **{a: t.user_attrs.get(a) for a in ATTRS}}
    for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
]).sort_values('mean_grid_dist').reset_index(drop=True)

print(f'{len(results_df)} completed trials')
results_df.head(20)"""),
("code", """\
# ── Pareto front ─────────────────────────────────────────────────────────────
pareto_numbers = {t.number for t in study.best_trials}
is_pareto = results_df['trial'].isin(pareto_numbers)

fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(
    results_df['mean_grid_dist'], results_df['overconfident_error_rate'],
    c=results_df['exact_match_accuracy'], cmap='RdYlGn', vmin=0, vmax=1,
    alpha=0.65, s=60, edgecolors='grey', linewidths=0.3,
)
ax.scatter(
    results_df.loc[is_pareto, 'mean_grid_dist'],
    results_df.loc[is_pareto, 'overconfident_error_rate'],
    color='black', marker='*', s=250, zorder=5,
    label=f'Pareto front ({is_pareto.sum()} trials)',
)
plt.colorbar(sc, ax=ax, label='Exact-match accuracy')
ax.set_xlabel('Mean grid distance  (world-model accuracy, lower = better)')
ax.set_ylabel('Overconfident-error rate  (posterior miscalibration, lower = better)')
ax.set_title('Transition Network Pareto Front — accuracy vs posterior calibration')
ax.legend()
plt.tight_layout()
plt.show()"""),
("code", """\
# ── Export Pareto candidates for the validation notebook (hpo2) ─────────────
PARAM_KEYS = ['hidden_dim', 'heads', 'learning_rate', 'num_encoder_layers',
              'forward_expansion', 'p_seq', 'p_attn']
candidates = [{
    'trial': t.number,
    'params': {k: t.user_attrs[k] for k in PARAM_KEYS},
    'mean_grid_dist': t.values[0],
    'overconfident_error_rate': t.values[1],
    'reward_mse': t.user_attrs.get('reward_mse'),
    'goal_path_grid_dist': t.user_attrs.get('goal_path_grid_dist'),
    'exact_match_accuracy': t.user_attrs.get('exact_match_accuracy'),
} for t in study.best_trials]

with open(TRANSITION_PARETO_JSON, 'w') as f:
    json.dump(candidates, f, indent=2)
print(f'Wrote {len(candidates)} Pareto candidates -> {TRANSITION_PARETO_JSON}')"""),
("md", """\
## Next step

Run **`hpo2_validation.ipynb`**: the top Pareto candidates are re-evaluated
with **full agent runs** (multi-seed cumulative regret) to check that these
offline objectives actually predict downstream performance — the
objective-mismatch check — and the winner is frozen for the value-network
search (hpo3)."""),
]

# ═══════════════════════════════ Notebook 2 ═══════════════════════════════

NB2 = [
("md", """\
# HPO Stage 2 — Validation: do offline model metrics predict online performance?

Model-based RL suffers from **objective mismatch** (Lambert et al., 2020): a
world model can score well on prediction metrics yet plan poorly. Before
freezing a transition config, this notebook re-evaluates the Stage-1 Pareto
candidates with **full BTRL agent runs** and checks the correlation.

## Plan
1. **Load** the Pareto candidates from `hpo_artifacts/transition_pareto.json`
   and rank them; keep the top `TOP_K` (by `mean_grid_dist`, ties broken by
   `overconfident_error_rate`). Add the **base-config transition params** as a
   baseline row.
2. **Evaluate online** — for each candidate: apply its transition params to
   the base config (value network **fixed at base config** across all
   candidates, so comparisons isolate the transition model), run the full
   agent for `TOTAL_STEPS` steps × `N_SEEDS` seeds, record cumulative regret,
   first-goal episode, episodes-to-solve.
3. **Correlate** — Spearman rank correlation between each offline objective
   and online cumulative regret, with scatter plots. Positive strong
   correlation ⇒ the offline objectives are valid proxies (thesis figure);
   weak/no correlation ⇒ evidence that transition tuning needs online signal.
4. **Freeze the winner** — lowest mean cumulative regret →
   `hpo_artifacts/best_transition_config.yaml`, consumed by hpo3.

Note on "steps to reach the end state": DeepSea episodes always last exactly
N steps, so within-episode path length is constant. The quantity that varies —
and that regret/episodes-to-solve measure — is how much *training experience*
the agent needs before it reliably reaches the goal (sample efficiency)."""),
("code", """\
import warnings
warnings.filterwarnings('ignore')

import json
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import spearmanr

from TRL.tuning.common import (
    TRANSITION_PARETO_JSON, VALIDATION_RESULTS_CSV, BEST_TRANSITION_CONFIG,
    load_base_config, save_config,
)
from TRL.tuning.transition_search import apply_transition_params
from TRL.tuning.agent_eval import evaluate_config

pd.options.display.float_format = '{:.4f}'.format"""),
("code", """\
# ── Budget knobs ─────────────────────────────────────────────────────────────
ENV          = '5'
TOP_K        = 5            # Pareto candidates to validate online
SEEDS        = [0, 1, 2]    # seeds per candidate
TOTAL_STEPS  = 10_000       # env steps per seed (~1.5 min/seed on GPU)

with open(TRANSITION_PARETO_JSON) as f:
    candidates = json.load(f)

candidates = sorted(
    candidates, key=lambda c: (c['mean_grid_dist'], c['overconfident_error_rate'])
)[:TOP_K]

# Baseline: the hand-set transition params from the base config.
base = load_base_config(env=ENV)
baseline_params = {k: base['transition'][k] for k in
                   ['hidden_dim', 'heads', 'learning_rate', 'num_encoder_layers',
                    'forward_expansion', 'p_seq', 'p_attn']}
candidates.append({'trial': 'baseline', 'params': baseline_params,
                   'mean_grid_dist': np.nan, 'overconfident_error_rate': np.nan,
                   'reward_mse': np.nan, 'goal_path_grid_dist': np.nan,
                   'exact_match_accuracy': np.nan})

print(f'{len(candidates)} configs x {len(SEEDS)} seeds x {TOTAL_STEPS} steps')
pd.DataFrame([{'trial': c['trial'], **c['params']} for c in candidates])"""),
("code", """\
# ── Full-agent evaluation (the expensive cell) ───────────────────────────────
rows = []
for i, cand in enumerate(candidates):
    config = apply_transition_params(load_base_config(env=ENV), cand['params'])
    print(f'[{datetime.now():%H:%M:%S}] {i+1}/{len(candidates)} trial={cand["trial"]} ...',
          end=' ', flush=True)
    summary = evaluate_config(config, SEEDS, TOTAL_STEPS)
    print(f'regret={summary["cumulative_regret_mean"]:.1f}'
          f'±{summary["cumulative_regret_std"]:.1f}')
    rows.append({
        'trial': cand['trial'], **cand['params'],
        'mean_grid_dist': cand['mean_grid_dist'],
        'overconfident_error_rate': cand['overconfident_error_rate'],
        'reward_mse': cand.get('reward_mse'),
        'goal_path_grid_dist': cand.get('goal_path_grid_dist'),
        'cumulative_regret_mean': summary['cumulative_regret_mean'],
        'cumulative_regret_std': summary['cumulative_regret_std'],
        'first_goal_episode_mean': summary['first_goal_episode_mean'],
        'episodes_to_solve_mean': summary['episodes_to_solve_mean'],
        'solved_fraction': summary['solved_fraction'],
        'solved_rate_last_100_mean': summary['solved_rate_last_100_mean'],
    })

validation_df = pd.DataFrame(rows).sort_values('cumulative_regret_mean').reset_index(drop=True)
validation_df.to_csv(VALIDATION_RESULTS_CSV, index=False)
print(f'\\nSaved -> {VALIDATION_RESULTS_CSV}')
validation_df"""),
("code", """\
# ── Objective-mismatch check: offline metrics vs online regret ──────────────
pareto_only = validation_df[validation_df['trial'] != 'baseline']
offline_metrics = ['mean_grid_dist', 'overconfident_error_rate',
                   'reward_mse', 'goal_path_grid_dist']

fig, axes = plt.subplots(1, len(offline_metrics), figsize=(5 * len(offline_metrics), 4.2))
for ax, metric in zip(np.atleast_1d(axes), offline_metrics):
    x = pareto_only[metric]
    y = pareto_only['cumulative_regret_mean']
    rho, p = spearmanr(x, y) if len(pareto_only) > 2 else (np.nan, np.nan)
    ax.errorbar(x, y, yerr=pareto_only['cumulative_regret_std'],
                fmt='o', capsize=3, markersize=8)
    for _, row in pareto_only.iterrows():
        ax.annotate(str(row['trial']), (row[metric], row['cumulative_regret_mean']),
                    textcoords='offset points', xytext=(6, 4), fontsize=8)
    ax.set_xlabel(f'{metric} (offline)')
    ax.set_ylabel('Cumulative regret (online)')
    ax.set_title(f'Spearman ρ={rho:.2f} (p={p:.2f})')
fig.suptitle('Objective-mismatch check: offline world-model metrics vs downstream regret')
plt.tight_layout()
plt.show()

print('Strong positive ρ  -> offline objective is a valid proxy for downstream performance.')
print('Weak/negative ρ    -> objective mismatch; transition tuning needs online signal.')"""),
("code", """\
# ── Freeze the winner for the value-network search ───────────────────────────
winner = validation_df.iloc[0]
winner_params = {k: winner[k] for k in ['hidden_dim', 'heads', 'learning_rate',
                                        'num_encoder_layers', 'forward_expansion',
                                        'p_seq', 'p_attn']}
winner_params = {k: (float(v) if k in ('learning_rate', 'p_seq', 'p_attn') else int(v))
                 for k, v in winner_params.items()}

best_config = apply_transition_params(load_base_config(env=ENV), winner_params)
save_config(best_config, BEST_TRANSITION_CONFIG)
print(f'Winner: trial={winner["trial"]} '
      f'(regret {winner["cumulative_regret_mean"]:.1f}±{winner["cumulative_regret_std"]:.1f})')
print(f'Frozen -> {BEST_TRANSITION_CONFIG}')
winner_params"""),
("md", """\
## Next step

Run **`hpo3_value_search.ipynb`**: with the transition network frozen at the
winning config, tune the value-network hyperparameters online against
seed-averaged cumulative regret.

If the winner was `baseline`, none of the searched configs beat the hand-set
params online — inspect the correlation plots before proceeding."""),
]

# ═══════════════════════════════ Notebook 3 ═══════════════════════════════

NB3 = [
("md", """\
# HPO Stage 3 — Value Network (online, single-objective)

The transition network is **frozen** at the validated winner
(`hpo_artifacts/best_transition_config.yaml`). Each trial runs the full BTRL
agent and is scored by **seed-averaged cumulative regret** — the
sample-efficiency metric of the PSDRL line of work. Regret integrates both
*whether* the agent learns to reach the goal and *how quickly*, so it captures
goal-reaching ability and sample efficiency in one number (in DeepSea they are
the same phenomenon: episodes are fixed-length, so only the amount of training
experience needed varies).

## Plan
1. **Setup** — load the frozen transition config; fall back to the base config
   with a loud warning if hpo2 has not been run.
2. **Objective** — per trial: sample value-network hyperparameters, run the
   full agent for `TOTAL_STEPS` × `N_SEEDS` seeds, return mean cumulative
   regret (minimise). Single-objective TPE — regret is the one number that
   matters here, and single-objective search also unlocks Optuna's
   param-importance analysis. No pruner: multi-seed RL scores are only
   comparable at full budget.
3. **Search** — persisted to the shared SQLite DB; resumable.
4. **Analysis** — trial table + hyperparameter importances.
5. **Export** — best full config → `hpo_artifacts/best_value_config.yaml`.
6. **Final comparison** — tuned vs base value config on *fresh held-out
   seeds*, with cumulative-regret learning curves.

**`discount` (γ) is deliberately not tuned**: DeepSea episodes are a fixed
N-step chain with a single terminal reward, so γ rescales all bootstrapped
values by the same monotonic factor and cannot change the greedy action
ranking. It stays at the base-config value."""),
("code", """\
import warnings
warnings.filterwarnings('ignore')

from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd
import optuna
from matplotlib import pyplot as plt

from TRL.tuning.common import (
    ARTIFACTS_DIR, OPTUNA_DB, VALUE_STUDY_NAME,
    BEST_TRANSITION_CONFIG, BEST_VALUE_CONFIG,
    load_base_config, load_config, save_config,
)
from TRL.tuning import value_search as vs
from TRL.tuning.agent_eval import evaluate_config, run_agent_trial

optuna.logging.set_verbosity(optuna.logging.WARNING)
pd.options.display.float_format = '{:.4f}'.format"""),
("code", """\
# ── Budget knobs ─────────────────────────────────────────────────────────────
SEED         = 42
N_TRIALS     = 50
SEEDS        = [0, 1, 2]   # training seeds per trial
TOTAL_STEPS  = 10_000      # env steps per seed (~1.5 min/seed on GPU)
HOLDOUT_SEEDS = [10, 11, 12]  # fresh seeds for the final tuned-vs-base check

if BEST_TRANSITION_CONFIG.exists():
    frozen_config = load_config(BEST_TRANSITION_CONFIG)
    print(f'Frozen transition config: {BEST_TRANSITION_CONFIG}')
else:
    frozen_config = load_base_config(env='5')
    print('WARNING: best_transition_config.yaml not found — run hpo2 first. '
          'Falling back to the BASE config transition params.')

print('transition:', {k: frozen_config['transition'][k] for k in
      ['hidden_dim', 'heads', 'learning_rate', 'num_encoder_layers',
       'forward_expansion', 'p_seq', 'p_attn']})
print('value (base, pre-tuning):', frozen_config['value'])"""),
("md", """\
## Search space

| Parameter | Choices | Represents |
|---|---|---|
| `training_iterations` (κ) | {5, 10, 20, 50} | value-net gradient steps per agent update — how hard the value function chases each newly sampled M̂ |
| `hidden_layers` | {1, 2, 3, 4} | MLP depth |
| `hidden_dim` | {8 … 256} | MLP width |
| `learning_rate` | log-uniform [1e-4, 1e-2] | Adam step size |
| `target_update_freq` | {5, 10, 20, 50} | steps between target-network copies — bootstrap-target stability |

Excluded: `discount` (see header)."""),
("code", """\
def objective(trial):
    params = vs.suggest_value_params(trial)
    config = vs.apply_value_params(deepcopy(frozen_config), params)

    print(f'[{datetime.now():%H:%M:%S}] trial {trial.number}: {params}', flush=True)
    summary = evaluate_config(config, SEEDS, TOTAL_STEPS)

    for key in ['cumulative_regret_mean', 'cumulative_regret_std',
                'first_goal_episode_mean', 'episodes_to_solve_mean',
                'solved_fraction', 'solved_rate_last_100_mean']:
        trial.set_user_attr(key, summary[key])
    return summary['cumulative_regret_mean']"""),
("code", """\
ARTIFACTS_DIR.mkdir(exist_ok=True)
study = optuna.create_study(
    study_name=VALUE_STUDY_NAME,
    storage=OPTUNA_DB,
    direction='minimize',
    sampler=optuna.samplers.TPESampler(seed=SEED),
    load_if_exists=True,
)

done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
remaining = max(0, N_TRIALS - done)
print(f'{done} trials already in the study; running {remaining} more '
      f'(~{remaining * len(SEEDS) * TOTAL_STEPS / 10_000 * 1.5:.0f} min on GPU)')
if remaining:
    study.optimize(objective, n_trials=remaining, show_progress_bar=True)
print(f'\\nBest trial: {study.best_trial.number} '
      f'regret={study.best_value:.1f} params={study.best_params}')"""),
("code", """\
# ── Trial table ──────────────────────────────────────────────────────────────
results_df = pd.DataFrame([
    {'trial': t.number, **t.params,
     'regret_mean': t.value,
     'regret_std': t.user_attrs.get('cumulative_regret_std'),
     'episodes_to_solve': t.user_attrs.get('episodes_to_solve_mean'),
     'first_goal_ep': t.user_attrs.get('first_goal_episode_mean'),
     'solved_fraction': t.user_attrs.get('solved_fraction')}
    for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
]).sort_values('regret_mean').reset_index(drop=True)
results_df.head(15)"""),
("code", """\
# ── Hyperparameter importances (fANOVA-style, single objective) ─────────────
importances = optuna.importance.get_param_importances(study)
fig, ax = plt.subplots(figsize=(8, 4))
names = list(importances)[::-1]
ax.barh(names, [importances[n] for n in names])
ax.set_xlabel('Importance for cumulative regret')
ax.set_title('Value-network hyperparameter importances')
plt.tight_layout()
plt.show()
importances"""),
("code", """\
# ── Export the fully tuned config ────────────────────────────────────────────
best_config = vs.apply_value_params(deepcopy(frozen_config), study.best_params)
save_config(best_config, BEST_VALUE_CONFIG)
print(f'Tuned config -> {BEST_VALUE_CONFIG}')
print('value:', best_config['value'])"""),
("code", """\
# ── Final check: tuned vs base value config on fresh held-out seeds ─────────
def regret_curves(config, seeds):
    curves = []
    for seed in seeds:
        res = run_agent_trial(config, seed, TOTAL_STEPS)
        curves.append(np.cumsum(res['optimal_return'] - np.array(res['episode_returns'])))
    return curves

print('Running held-out comparison (~%.0f min)...'
      % (2 * len(HOLDOUT_SEEDS) * TOTAL_STEPS / 10_000 * 1.5))
tuned_curves = regret_curves(best_config, HOLDOUT_SEEDS)
base_curves = regret_curves(frozen_config, HOLDOUT_SEEDS)

fig, ax = plt.subplots(figsize=(9, 5))
for label, curves, color in [('tuned', tuned_curves, 'tab:green'),
                             ('base', base_curves, 'tab:red')]:
    n = min(len(c) for c in curves)
    stack = np.stack([c[:n] for c in curves])
    ax.plot(stack.mean(0), color=color, label=f'{label} value config')
    ax.fill_between(range(n), stack.min(0), stack.max(0), color=color, alpha=0.15)
ax.set_xlabel('Episode')
ax.set_ylabel('Cumulative regret')
ax.set_title(f'Tuned vs base value network — held-out seeds {HOLDOUT_SEEDS}\\n'
             '(flatter = solved; earlier bend = more sample-efficient)')
ax.legend()
plt.tight_layout()
plt.show()

print(f'Final regret  tuned: {np.mean([c[-1] for c in tuned_curves]):.1f}   '
      f'base: {np.mean([c[-1] for c in base_curves]):.1f}')"""),
]

if __name__ == "__main__":
    build("hpo1_transition_search.ipynb", NB1)
    build("hpo2_validation.ipynb", NB2)
    build("hpo3_value_search.ipynb", NB3)
