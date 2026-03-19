import math


def reservation_price(
    S: float,
    q: int,
    gamma: float,
    sigma: float,
    T: float,
    t: float,
) -> float:
    """
    r_t = S_t - q_t * gamma * sigma^2 * (T - t)
    """
    time_left = max(T - t, 0.0)
    return S - q * gamma * (sigma ** 2) * time_left


def optimal_half_spread(
    gamma: float,
    sigma: float,
    T: float,
    t: float,
    k: float,
) -> float:
    """
    delta_t = (1/gamma) * ln(1 + gamma/k) + 0.5 * gamma * sigma^2 * (T - t)

    Assumes gamma > 0 and k > 0.
    """
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    if k <= 0:
        raise ValueError("k must be > 0")

    time_left = max(T - t, 0.0)
    return (1.0 / gamma) * math.log(1.0 + gamma / k) + 0.5 * gamma * (sigma ** 2) * time_left


def compute_quotes(
    S: float,
    q: int,
    gamma: float,
    sigma: float,
    T: float,
    t: float,
    k: float,
) -> tuple[float, float, float, float]:
    """
    Returns:
        reservation_price, half_spread, bid, ask
    """
    r = reservation_price(S=S, q=q, gamma=gamma, sigma=sigma, T=T, t=t)
    delta = optimal_half_spread(gamma=gamma, sigma=sigma, T=T, t=t, k=k)
    bid = r - delta
    ask = r + delta
    return r, delta, bid, ask