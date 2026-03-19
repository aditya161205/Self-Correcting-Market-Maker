import numpy as np


def next_regime(current_regime: int, p00: float, p11: float, rng: np.random.Generator) -> int:
    """
    Two-state Markov chain:
        Regime 0 -> stay with prob p00
        Regime 1 -> stay with prob p11
    """
    u = rng.random()

    if current_regime == 0:
        return 0 if u < p00 else 1
    else:
        return 1 if u < p11 else 0


def regime_parameters(
    regime: int,
    sigma_0: float,
    sigma_1: float,
    mu_0: float,
    mu_1: float,
) -> tuple[float, float]:
    if regime == 0:
        return mu_0, sigma_0
    return mu_1, sigma_1


def step_price_regime_switching(
    S: float,
    mu: float,
    sigma: float,
    dt: float,
    rng: np.random.Generator,
) -> float:
    """
    Markov regime-switching price update:
        S_{t+dt} = S_t + mu * dt + sigma * sqrt(dt) * Z
    """
    z = rng.normal()
    return S + mu * dt + sigma * np.sqrt(dt) * z