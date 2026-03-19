import math
import numpy as np


def intensity(A: float, k: float, distance: float) -> float:
    """
    lambda = A * exp(-k * distance)

    distance should ideally be >= 0, but we clamp below at 0
    to avoid exploding intensities if a quote crosses mid.
    """
    d = max(distance, 0.0)
    return A * math.exp(-k * d)


def fill_probability(lmbda: float, dt: float) -> float:
    """
    P(fill in dt) = 1 - exp(-lambda * dt)
    """
    p = 1.0 - math.exp(-lmbda * dt)
    return min(max(p, 0.0), 1.0)


def simulate_fill(prob: float, rng: np.random.Generator) -> bool:
    return rng.random() < prob


def bid_ask_fill_probabilities(
    S: float,
    bid: float,
    ask: float,
    A: float,
    k: float,
    dt: float,
) -> tuple[float, float, float, float]:
    """
    Returns:
        lambda_bid, lambda_ask, prob_bid_fill, prob_ask_fill
    """
    bid_distance = S - bid
    ask_distance = ask - S

    lambda_bid = intensity(A=A, k=k, distance=bid_distance)
    lambda_ask = intensity(A=A, k=k, distance=ask_distance)

    p_bid = fill_probability(lambda_bid, dt)
    p_ask = fill_probability(lambda_ask, dt)

    return lambda_bid, lambda_ask, p_bid, p_ask