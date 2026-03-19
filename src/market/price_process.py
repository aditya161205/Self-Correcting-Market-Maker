import numpy as np


def step_price(S: float, sigma: float, dt: float, rng: np.random.Generator) -> float:
    """
    One-step Brownian motion update:
        S_{t+dt} = S_t + sigma * sqrt(dt) * Z,   Z ~ N(0,1)
    """
    z = rng.normal()
    return S + sigma * np.sqrt(dt) * z