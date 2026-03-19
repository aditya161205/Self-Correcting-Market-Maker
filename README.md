# Self-Correcting Market Maker (SCMM)

A research-oriented market making project that starts from the classical **Avellaneda–Stoikov (AS)** model and extends it into a **gated, risk-aware, self-correcting controller** that performs better than AS under stressful market conditions.

## Project overview

The main idea of this project is simple:

- use **Avellaneda–Stoikov** as the mathematically grounded baseline
- add a **self-correcting control layer** on top of it
- evaluate both models under increasingly difficult environments
- identify whether adaptive control improves profitability, survivability, and risk-adjusted performance

Instead of replacing AS with a black-box policy, the project preserves the AS structure and learns or designs **corrections** around it.

The final successful controller in this repository is **SCMM-v4**, a **gated, risk-aware, rule-based market maker**.

---

## Core idea

The classical Avellaneda–Stoikov model computes:

- a reservation price
- an optimal half-spread
- bid and ask quotes around that center

In this project, SCMM augments that structure with two control signals:

- **skew correction**: adjusts the quote center
- **spread correction**: adjusts the half-spread

Formally,

\[
r_t^{SC} = r_t^{AS} + u_t^{skew}
\]

\[
\delta_t^{SC} = \delta_t^{AS} + u_t^{spr}
\]

and the final bid and ask become

\[
p_t^b = r_t^{SC} - \delta_t^{SC}, \qquad p_t^a = r_t^{SC} + \delta_t^{SC}
\]

This keeps the strong AS prior while allowing the controller to respond to:
- inventory stress
- drawdown
- realized volatility
- time-to-maturity
- stressed market conditions

---

## What has been built

### 1. Classical Avellaneda–Stoikov baseline
The project includes a full baseline simulator with:
- Brownian mid-price simulation
- exponential fill intensity model
- stochastic fills
- inventory and cash bookkeeping
- marked-to-market wealth tracking
- plotting and multi-run evaluation

### 2. Risk-aware evaluation framework
A reusable evaluation pipeline was built to compare models on:
- final wealth
- step-level Sharpe-like ratio
- drawdown metrics
- inventory metrics
- fill imbalance
- quote stability
- survival under capital constraints

### 3. Rule-based SCMM variants
Several SCMM variants were tested:
- inventory-only correction
- inventory + drawdown correction
- richer state-aware correction
- gated SCMM-v4

The winning design was **SCMM-v4**, which uses:
- **inventory-aware skew**
- **drawdown-aware spread widening**
- **realized-volatility-aware spread widening**
- **time urgency near terminal horizon**
- **gating**, so corrections are activated only when needed

### 4. RL experiments
Gymnasium + Stable-Baselines3 PPO environments were built for RL-based correction policies.
However, the PPO variants did **not** outperform the best rule-based controller.
This is an important negative result of the project:
- structured, gated control outperformed the RL variants tested so far

### 5. Markov regime-switching environment
A two-state **Markov regime-switching market environment** was built:
- **Regime 0**: calm market
- **Regime 1**: toxic market

The regime evolves using a two-state Markov chain, with regime-dependent:
- volatility
- drift

This allows the project to test whether a self-correcting controller is actually useful when market conditions deteriorate.

### 6. Absorbing barrier / capital constraint
An absorbing drawdown barrier was added:
- if drawdown exceeds a threshold, the episode terminates
- this lets us measure:
  - barrier-hit rate
  - survival rate
  - time-to-ruin

This makes the evaluation much more realistic from a risk-management perspective.

### 7. Ablation suite
A full ablation suite was run for SCMM-v4, testing:
- no gating
- no drawdown term
- no volatility term
- no time urgency

These experiments showed that the full SCMM-v4 outperformed all tested ablations in the stressed regime + barrier setting.

---

## Final model: SCMM-v4

The best-performing model in this repository is:

### **Gated SCMM-v4**

It is a **rule-based**, **risk-aware**, **state-dependent** controller built on top of Avellaneda–Stoikov.

The key idea is:
- remain close to AS in normal conditions
- activate corrections only in stressed states

The model uses:
- moderate inventory skew correction
- light spread widening under drawdown and volatility stress
- time-scaled urgency near the end of the episode
- gating to avoid unnecessary over-correction

### Winning parameter set
The best controller found through structured parameter search used:

- `scmm_alpha_q = 0.10`
- `scmm_alpha_f = 0.00`
- `scmm_beta_q = 0.05`
- `scmm_beta_d = 0.01`
- `scmm_beta_sigma = 0.20`
- `scmm_eta_time = 0.75`

with gating enabled.

---

## Main findings

### Stationary environment
In the stationary environment, SCMM-v4 improved:
- Sharpe-like ratio
- drawdown
- inventory exposure
- fill balance

while preserving essentially the same profitability as AS.

### Markov regime-switching environment
In the regime-switching environment, SCMM-v4 outperformed AS more clearly:
- higher wealth
- better Sharpe-like performance
- lower inventory risk
- lower drawdown
- better downside tail behavior

### Regime-switching + absorbing barrier
This is the strongest result in the project.

Under regime switching with a hard drawdown barrier, SCMM-v4 produced:
- higher mean wealth
- better Sharpe-like performance
- lower barrier-hit rate
- higher survival rate
- longer mean survival time

### 1,000-run significance study
In a large-sample 1,000-run comparison, SCMM-v4 significantly outperformed AS in the regime + barrier setting.

The project found statistically significant improvements in:
- final wealth
- Sharpe-like ratio
- mean absolute inventory
- maximum drawdown
- steps survived
- barrier-hit rate

### Time-to-ruin analysis
As the episode horizon increases, both AS and SCMM-v4 eventually approach high ruin probability under finite capital constraints.

However, SCMM-v4 exhibits a superior **time-to-ruin** profile:
- higher survival at moderate horizons
- lower early barrier-hit frequency
- longer mean survival time

This project therefore does **not** claim asymptotic immunity to ruin.
Instead, it shows that SCMM-v4 **delays ruin and improves survivability** in toxic, capital-constrained environments.

---

## Repository structure

```text
scmm_project/
├── README.md
├── requirements.txt
├── config/
│   └── default_params.py
├── src/
│   ├── market/
│   │   ├── price_process.py
│   │   └── regime_switching.py
│   ├── policy/
│   │   ├── avellaneda_stoikov.py
│   │   └── scmm_rule.py
│   ├── execution/
│   │   └── fill_model.py
│   ├── portfolio/
│   │   └── portfolio.py
│   ├── simulator/
│   │   ├── simulator.py
│   │   ├── scmm_v4_gated_simulator.py
│   │   ├── as_regime_simulator.py
│   │   ├── scmm_v4_gated_regime_simulator.py
│   │   ├── as_regime_barrier_simulator.py
│   │   └── scmm_v4_gated_regime_barrier_simulator.py
│   ├── metrics/
│   │   └── performance.py
│   ├── rl/
│   │   ├── env_scmm.py
│   │   ├── env_scmm_v5.py
│   │   ├── train_ppo.py
│   │   └── train_ppo_v5.py
│   └── utils/
│       ├── logger.py
│       └── helpers.py
├── experiments/
│   ├── run_baseline.py
│   ├── run_multi_baseline.py
│   ├── run_multi_scmm_v4_gated.py
│   ├── run_regime_comparison.py
│   ├── run_regime_barrier_comparison.py
│   ├── run_regime_barrier_significance.py
│   ├── run_long_horizon_ruin_test.py
│   ├── run_survival_curves.py
│   └── run_scmm_v4_ablation_suite.py
├── outputs/
│   ├── logs/
│   ├── plots/
│   └── results/
└── paper_draft.md