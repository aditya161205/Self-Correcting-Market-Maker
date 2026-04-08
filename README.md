# Self-Correcting Market Maker (SCMM)

A research-oriented market making project that starts from the classical **Avellaneda–Stoikov (AS)** framework and extends it into a **gated, risk-aware, self-correcting controller** that performs better than vanilla AS under stressful market conditions.

---

## Overview

This project studies whether a market maker can be improved by adding a **selective correction layer** on top of the classical Avellaneda–Stoikov model.

Instead of replacing AS with a black-box strategy, the project preserves the AS structure and adds adaptive corrections to:
- quote skew
- spread width
- risk response under stress

The final successful controller in this repository is **SCMM-v4**, a **gated, risk-aware, rule-based self-correcting market maker**.

The project also includes:
- a validated AS baseline
- a reusable evaluation framework
- regime-switching toxic-market environments
- an absorbing drawdown barrier
- time-to-ruin analysis
- ablation studies
- exploratory RL extensions

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

$$r_t^{SC} = r_t^{AS} + u_t^{\text{skew}}$$

$$\delta_t^{SC} = \delta_t^{AS} + u_t^{\text{spr}}$$

and the final bid and ask become

$$p_t^b = r_t^{SC} - \delta_t^{SC}, \qquad p_t^a = r_t^{SC} + \delta_t^{SC}$$

This keeps the strong AS prior while allowing the controller to respond to:
- inventory stress
- drawdown
- realized volatility
- time-to-maturity
- stressed market conditions

So the main philosophy of the project is:

> **Do not abandon AS. Correct it when needed.**

---

## Baseline: Avellaneda–Stoikov

The classical AS model uses:

### Reservation price

$$r_t = S_t - q_t \gamma \sigma^2 (T-t)$$

### Half-spread

$$\delta_t^{AS} = \frac{1}{\gamma}\ln\left(1+\frac{\gamma}{k}\right) + \frac{1}{2}\gamma \sigma^2 (T-t)$$

### Quotes

$$p_t^b = r_t - \delta_t^{AS}, \qquad p_t^a = r_t + \delta_t^{AS}$$

The project implements this baseline fully with:
- Brownian mid-price simulation
- exponential fill-intensity model
- stochastic bid/ask fills
- inventory and cash bookkeeping
- marked-to-market wealth
- multi-run validation and evaluation

---

## What SCMM adds

Several self-correcting controllers were tested, beginning from simple inventory-only feedback and progressively adding more state-awareness.

The final successful model is:

## **SCMM-v4**
A **gated**, **risk-aware**, **state-dependent** controller built on top of AS.

Its main ingredients are:
- inventory-aware skew correction
- drawdown-aware spread widening
- realized-volatility-aware spread widening
- time urgency near the end of the episode
- gating so that correction is activated only when needed

### Mathematical Formulation of SCMM-v4

SCMM-v4 utilizes logical gates to ensure that corrections are only applied when specific risk thresholds are breached. Let the indicator functions for these gates be:

$$I_q = \mathbb{1}(|q_t| \ge q_{\text{threshold}})$$

$$I_f = \mathbb{1}(|\text{fill imbalance}_t| \ge \text{imbalance}_{\text{threshold}})$$

$$I_d = \mathbb{1}(\text{drawdown}_t \ge \text{drawdown}_{\text{threshold}})$$

The time urgency multiplier increases linearly as the episode nears its end:

$$\tau_t = 1.0 + \eta_{\text{time}} \frac{t}{T}$$

The final skew and spread corrections applied to the AS prior are calculated as:

$$u_t^{\text{skew}} = I_q (-\alpha_q q_t \tau_t) + I_f (-\alpha_f \cdot \text{fill imbalance}_t)$$

$$u_t^{\text{spr}} = \beta_q |q_t| + I_d (\beta_d \cdot \text{drawdown}_t) + \beta_\sigma \cdot \text{realized vol}_t$$

The winning SCMM-v4 is therefore **not** the most aggressive model.  
It is a **lightly corrective, selectively adaptive** controller.

---

## Project components

This repository contains the following major pieces.

### 1. AS baseline simulator
A full implementation of the classical Avellaneda–Stoikov model.

### 2. Risk-aware evaluation framework
A reusable metrics module for:
- final wealth
- Sharpe-like ratio
- maximum drawdown
- mean drawdown
- inventory statistics
- fill imbalance
- quote stability
- survival metrics

### 3. Rule-based SCMM variants
Several SCMM variants were developed and tested:
- inventory-only
- inventory + drawdown
- richer state-aware control
- gated SCMM-v4

### 4. RL extensions
Gymnasium + Stable-Baselines3 PPO environments were built for RL-based correction policies.  
These RL variants did **not** outperform the best rule-based controller.

### 5. Markov regime-switching environment
A 2-state **Markov regime-switching** market environment was built:
- **Regime 0**: calm market
- **Regime 1**: toxic market

The regime evolves using a hidden Markov chain with regime-dependent:
- drift
- volatility

### 6. Absorbing drawdown barrier
A hard drawdown barrier was introduced:
- if drawdown exceeds a threshold, the episode terminates
- this enables survival analysis and time-to-ruin experiments

### 7. Ablation suite
A full ablation study was run for SCMM-v4 by removing:
- gating
- drawdown term
- volatility term
- time urgency

These ablations showed that the full SCMM-v4 outperformed all tested simplified variants.

---

## Final winning model

The best-performing controller in this repository is **gated SCMM-v4**.

### Winning parameter set
- `scmm_alpha_q = 0.10`
- `scmm_alpha_f = 0.00`
- `scmm_beta_q = 0.05`
- `scmm_beta_d = 0.01`
- `scmm_beta_sigma = 0.20`
- `scmm_eta_time = 0.75`

with gating enabled.

### Interpretation
This result is important:
- moderate inventory skew helped
- over-aggressive skew hurt
- fill-imbalance skew was not useful in the winning version
- light, selective spread widening worked better than heavy widening

---

## Quantitative results

The project was evaluated in progressively harder environments.

---

## 1. Stationary environment (50-run comparison)

### Avellaneda–Stoikov baseline
- mean final wealth: **22.5316**
- mean Sharpe-like: **0.1488**
- mean absolute inventory: **0.5042**
- mean max drawdown: **2.7125**
- mean fill imbalance: **1.48**

### Winning gated SCMM-v4
- mean final wealth: **22.5227**
- mean Sharpe-like: **0.1515**
- mean absolute inventory: **0.4598**
- mean max drawdown: **2.3064**
- mean fill imbalance: **1.18**

### Interpretation
In the stationary environment, SCMM-v4 preserves essentially the same profitability as AS while improving:
- risk-adjusted performance
- inventory stability
- drawdown
- fill balance

So even in benign conditions, SCMM-v4 is a stronger risk-aware controller than vanilla AS.

---

## 2. Markov regime-switching environment (50-run comparison)

### AS under regime switching
- mean final wealth: **19.4635**
- mean Sharpe-like: **0.1200**
- 5th percentile wealth: **6.4768**
- mean absolute inventory: **0.8800**
- mean max drawdown: **5.6385**
- mean drawdown: **1.7268**
- mean fill imbalance: **1.90**

### SCMM-v4 under regime switching
- mean final wealth: **20.1286**
- mean Sharpe-like: **0.1332**
- 5th percentile wealth: **8.5378**
- mean absolute inventory: **0.7172**
- mean max drawdown: **4.4673**
- mean drawdown: **1.2757**
- mean fill imbalance: **1.26**

### Interpretation
When the market switches between calm and toxic regimes, SCMM-v4 outperforms AS more clearly.  
It achieves:
- higher wealth
- better downside protection
- lower inventory risk
- lower drawdown
- better execution balance

This is where the self-correcting design becomes especially valuable.

---

## 3. Regime switching + absorbing drawdown barrier (1,000-run comparison)

This is the strongest result in the project.

### AS with barrier
- mean final wealth: **15.5349**
- mean Sharpe-like: **0.0986**
- 5th percentile wealth: **-5.7452**
- mean absolute inventory: **0.8301**
- mean max drawdown: **5.1813**
- barrier-hit rate: **35.1%**
- survival rate: **64.9%**
- mean steps survived: **401.1**

### SCMM-v4 with barrier
- mean final wealth: **16.8911**
- mean Sharpe-like: **0.1123**
- 5th percentile wealth: **-5.1558**
- mean absolute inventory: **0.7156**
- mean max drawdown: **4.4490**
- barrier-hit rate: **24.6%**
- survival rate: **75.4%**
- mean steps survived: **426.4**

### Improvement over AS
Compared with AS in the regime-switching barrier setting, SCMM-v4 delivers:
- **+1.356** higher mean final wealth
- **+0.0137** higher Sharpe-like score
- **0.1146 lower** mean absolute inventory
- **0.7323 lower** mean max drawdown
- **10.5 percentage points lower** barrier-hit rate
- **10.5 percentage points higher** survival rate
- **25.2 more steps survived on average**

### Statistical significance
The improvements are statistically significant:

- final wealth  
  - Welch t-test p-value: **0.0075**
  - Mann–Whitney p-value: **0.0231**

- Sharpe-like  
  - p-value ≈ **1.24e-4**

- mean absolute inventory  
  - p-value ≈ **1.79e-19**

- maximum drawdown  
  - p-value ≈ **5.34e-8**

- steps survived  
  - p-value ≈ **1.37e-4**

- barrier-hit rate  
  - Fisher exact test p-value: **3.55e-7**

### Interpretation
Under toxic regime switching and finite capital, SCMM-v4 is not just marginally better than AS.  
It is:
- more profitable
- more stable
- more survivable
- statistically significantly better on the major risk-aware metrics

---

## 4. Time-to-ruin analysis

The simulation horizon was expanded to test long-run failure behavior under a hard capital constraint.

### Survival rates across horizon lengths

| Horizon \(T\) | AS survival | SCMM-v4 survival |
|---|---:|---:|
| 5  | 68.67% | 78.00% |
| 10 | 26.00% | 31.67% |
| 20 | 1.67%  | 2.33%  |
| 40 | 0.33%  | 0.33%  |

### Interpretation
As the horizon becomes very large, both AS and SCMM-v4 eventually approach high ruin probability under finite capital constraints.

This is expected in a persistent Markov regime-switching environment with repeated toxic exposure.

The important result is therefore **time-to-ruin**, not asymptotic immunity to ruin.

SCMM-v4 consistently:
- delays barrier intersection
- improves survival probability at moderate horizons
- survives longer on average

So the project’s claim is not that SCMM-v4 survives forever, but that it is **structurally more robust** than AS over relevant operating horizons.

---

## Ablation results

A full ablation suite was run in the regime-switching + barrier environment.

### Models tested
- AS baseline
- full SCMM-v4
- SCMM-v4 without gating
- SCMM-v4 without drawdown term
- SCMM-v4 without volatility term
- SCMM-v4 without time urgency

### Key findings
The full SCMM-v4 outperformed all tested ablations.

Removing any one of the major components degraded performance.

In particular:
- **gating helps**
- **drawdown-aware correction helps**
- **volatility-aware spread control helps**
- **time urgency helps a lot**

This shows that SCMM-v4’s edge does not come from one arbitrary tweak.  
It comes from the combination of:
- selective activation
- risk awareness
- volatility adaptation
- terminal inventory urgency

---

## RL results

RL-based SCMM variants were explored using:
- Gymnasium
- Stable-Baselines3 PPO
- direct AS + RL correction
- SCMM-v4 + RL residual correction (SCMM-v5)

### SCMM-v5 Mathematical Formulation
SCMM-v5 is formulated as a residual correction algorithm layered on top of the SCMM-v4 prior. Instead of learning quotes entirely from scratch, the RL agent is tasked with learning continuous residual adjustments.

**State Space (Observation)**
The agent observes a 10-dimensional normalized state vector at each step:
1. Normalized Inventory: $q_t / 10.0$
2. Absolute Inventory: $|q_t| / 10.0$
3. Time Remaining: $(T - t) / T$
4. Drawdown: $\text{drawdown}_t / 10.0$
5. Fill Imbalance: $\text{fill imbalance}_t / 20.0$
6. Realized Volatility: $\text{realized vol}_t / 5.0$
7. Market Regime: $regime_t \in \{0, 1\}$
8. AS Half-Spread: $\delta_t^{AS} / 5.0$
9. SCMM-v4 Skew Correction: $u_t^{\text{skew, v4}} / 1.0$
10. SCMM-v4 Spread Correction: $u_t^{\text{spr, v4}} / 1.0$

**Action Space**
The policy outputs a continuous 2D action vector $a \in [-1, 1]^2$. These actions are scaled to generate the small residual corrections:

$$du_t^{\text{skew}} = a_0 \cdot \text{residual skew max}$$

$$du_t^{\text{spr}} = a_1 \cdot \text{residual spr max}$$

The final applied corrections update the underlying v4 prior:

$$u_t^{\text{skew}} = u_t^{\text{skew, v4}} + du_t^{\text{skew}}$$

$$u_t^{\text{spr}} = u_t^{\text{spr, v4}} + du_t^{\text{spr}}$$

**Reward Function**
The reward mechanism balances the standard P&L generation objective against harsh penalties for holding excessive inventory, suffering drawdowns, and over-relying on the residual actions (which encourages preserving the safety of the SCMM-v4 rule-base unless absolutely necessary):

$$R_t = \Delta \text{wealth}_t - \lambda_q q_t^2 - \lambda_d \cdot \text{drawdown}_t - \lambda_{\text{res}} \left( (du_t^{\text{skew}})^2 + (du_t^{\text{spr}})^2 \right)$$

These RL variants did **not** outperform the best rule-based controller.

This is an important result of the project:
- structured, interpretable, gated control outperformed the RL variants tested so far

So the current best model is still:

## **SCMM-v4**
not the RL extensions.

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
│   │   ├── scmm_v4_gated_regime_barrier_simulator.py
│   │   └── ...
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
│   ├── run_scmm_v4_ablation_suite.py
│   └── ...
├── outputs/
│   ├── logs/
│   ├── plots/
│   └── results/
└── paper_draft.md
