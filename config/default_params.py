from dataclasses import dataclass


@dataclass
class Params:
    S0: float = 100.0
    sigma: float = 2.0
    gamma: float = 0.1
    A: float = 8.0
    k: float = 1.0
    T: float = 5.0
    dt: float = 0.01
    seed: int = 42
    max_inventory: int | None = None

    # SCMM coefficients
    scmm_alpha_q: float = 0.10
    scmm_alpha_f: float = 0.00
    scmm_beta_q: float = 0.05
    scmm_beta_d: float = 0.01
    scmm_beta_sigma: float = 0.20
    scmm_beta_f: float = 0.00
    scmm_eta_time: float = 0.75

    # Realized vol config
    scmm_vol_window: int = 20
    scmm_vol_scale: float = 1.0

    # Gating thresholds
    scmm_q_threshold: int = 1
    scmm_drawdown_threshold: float = 1.0
    scmm_imbalance_threshold: float = 2.0

    # Markov regime-switching parameters
    regime_p00: float = 0.98
    regime_p11: float = 0.90

    regime_sigma_0: float = 1.0
    regime_sigma_1: float = 4.0

    regime_mu_0: float = 0.0
    regime_mu_1: float = -1.5

    # Absorbing Barrier 
    barrier_drawdown_limit: float = 6.0
    barrier_terminal_penalty: float = 0.0

    # Ablation flags
    scmm_use_gating: bool = True
    scmm_use_drawdown_term: bool = True
    scmm_use_vol_term: bool = True
    scmm_use_time_urgency: bool = True