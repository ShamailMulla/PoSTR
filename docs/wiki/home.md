# BTRL Thesis Knowledge Base

Bayesian Transformer Reinforcement Learning — research wiki for the MSc thesis project at Queen Mary University of London.

---

## Core Concepts

- [[bayesformer]] — Transformer with Bayesian uncertainty via structured dropout (Meta AI, 2022)
- [[psdrl]] — Posterior Sampling for Deep RL — the baseline this thesis extends
- [[btrl]] — This thesis: BayesFormer as the transition model inside PSDRL
- [[thompson-sampling]] — The exploration strategy underpinning PSDRL and BTRL
- [[variational-inference]] — Theoretical grounding for dropout-as-Bayes
- [[planning-in-sampled-model]] — **Proposal (2026-09):** drop the value net, plan exactly in each posterior sample — PSRL theory, regret, open proofs

## Architecture

- [[transition-network]] — Transformer world model: states → predicted next states
- [[episode-locked-dropout]] — The critical fix enabling Thompson Sampling with Transformers
- [[dropout-architecture]] — Two independent dropout families: p_seq (site i) and p_attn (sites ii/iii/iv)
- [[neural-linear-head]] — v3: BLR posterior replacing dropout (the fair comparison to PSDRL)
- [[value-network]] — MLP critic that estimates V(s) over sampled world models
- [[deepsea]] — The bsuite benchmark environment used for experiments

## Version History & Failure Modes

- [[failure-modes]] — **Start here.** The two-options narrative (BayesFormer vs PSDRL uncertainty), the version arc, and every failure + its diagnostic methodology
- [[dithering-trap]] — v1 bug: step-wise mask resampling destroys exploration → [[episode-locked-dropout]]; also scheduled vs per-step resampling, and the v3+ mid-episode redraw
- [[representation-drift]] — v1 bug: non-stationary hidden states prevent value learning → [[episode-locked-dropout]]
- [[neural-linear-head]] — Why dropout was set aside and BTRL returned to PSDRL's data-coupled posterior

## Implementation

- [[loss-functions]] — Environment-aware state transition loss (grid / categorical / continuous)
- [[optuna-objectives]] — The two Pareto objectives for transition model hyperparameter search
- [[psdrl-autoencoder-analysis]] — Why PSDRL uses a CNN AE for Atari and why it's removed for DeepSea (embed_dim=25 fixed)
- [[codebase]] — Module map and execution flow

## Meta

- [[learnings]] — Session-by-session log of what works and what doesn't
- [[dissertation-feedback]] — Examiner feedback and response

---

## Source Documents

| Document | Wiki Page |
|---|---|
| `BayesFormer.pdf` | [[bayesformer]] |
| `PSDRL-pages.pdf` | [[psdrl]] |
| `BTRL-architecture.pdf` | [[btrl]] |
| `integration_problem.txt` | [[dithering-trap]], [[representation-drift]], [[episode-locked-dropout]] |
| `Dissertation Feedback.txt` | [[dissertation-feedback]] |
| `Dissertation.pdf` | [[btrl]], [[codebase]] |
