from src.policy.avellaneda_stoikov import compute_quotes


def compute_time_urgency(t: float, T: float, eta_time: float) -> float:
    if T <= 0:
        return 1.0
    return 1.0 + eta_time * (t / T)


def compute_scmm_v3_corrections(
    q: int,
    drawdown: float,
    fill_imbalance: float,
    realized_vol: float,
    t: float,
    T: float,
    alpha_q: float,
    alpha_f: float,
    beta_q: float,
    beta_d: float,
    beta_sigma: float,
    beta_f: float,
    eta_time: float,
) -> tuple[float, float]:
    time_urgency = compute_time_urgency(t=t, T=T, eta_time=eta_time)

    u_skew = -alpha_q * q * time_urgency - alpha_f * fill_imbalance
    u_spr = (
        beta_q * abs(q)
        + beta_d * drawdown
        + beta_sigma * realized_vol
        + beta_f * abs(fill_imbalance)
    )
    return u_skew, u_spr


def compute_scmm_v3_quotes(
    S: float,
    q: int,
    drawdown: float,
    fill_imbalance: float,
    realized_vol: float,
    gamma: float,
    sigma: float,
    T: float,
    t: float,
    k: float,
    alpha_q: float,
    alpha_f: float,
    beta_q: float,
    beta_d: float,
    beta_sigma: float,
    beta_f: float,
    eta_time: float,
) -> tuple[float, float, float, float, float, float, float, float]:
    r_as, delta_as, _, _ = compute_quotes(
        S=S,
        q=q,
        gamma=gamma,
        sigma=sigma,
        T=T,
        t=t,
        k=k,
    )

    u_skew, u_spr = compute_scmm_v3_corrections(
        q=q,
        drawdown=drawdown,
        fill_imbalance=fill_imbalance,
        realized_vol=realized_vol,
        t=t,
        T=T,
        alpha_q=alpha_q,
        alpha_f=alpha_f,
        beta_q=beta_q,
        beta_d=beta_d,
        beta_sigma=beta_sigma,
        beta_f=beta_f,
        eta_time=eta_time,
    )

    r_sc = r_as + u_skew
    delta_sc = max(delta_as + u_spr, 1e-6)

    bid_sc = r_sc - delta_sc
    ask_sc = r_sc + delta_sc

    return r_as, delta_as, r_sc, delta_sc, u_skew, u_spr, bid_sc, ask_sc


def compute_scmm_v4_gated_corrections(
    q: int,
    drawdown: float,
    fill_imbalance: float,
    realized_vol: float,
    t: float,
    T: float,
    alpha_q: float,
    alpha_f: float,
    beta_q: float,
    beta_d: float,
    beta_sigma: float,
    eta_time: float,
    q_threshold: int,
    drawdown_threshold: float,
    imbalance_threshold: float,
    use_gating: bool = True,
    use_drawdown_term: bool = True,
    use_vol_term: bool = True,
    use_time_urgency: bool = True,
) -> tuple[float, float]:
    """
    Gated SCMM-v4 with ablation switches.
    """
    time_urgency = compute_time_urgency(t=t, T=T, eta_time=eta_time) if use_time_urgency else 1.0

    if use_gating:
        inventory_gate = 1.0 if abs(q) >= q_threshold else 0.0
        imbalance_gate = 1.0 if abs(fill_imbalance) >= imbalance_threshold else 0.0
        drawdown_gate = 1.0 if drawdown >= drawdown_threshold else 0.0
    else:
        inventory_gate = 1.0
        imbalance_gate = 1.0
        drawdown_gate = 1.0

    drawdown_component = drawdown_gate * beta_d * drawdown if use_drawdown_term else 0.0
    vol_component = beta_sigma * realized_vol if use_vol_term else 0.0

    u_skew = (
        inventory_gate * (-alpha_q * q * time_urgency)
        + imbalance_gate * (-alpha_f * fill_imbalance)
    )

    u_spr = (
        beta_q * abs(q)
        + drawdown_component
        + vol_component
    )

    return u_skew, u_spr


def compute_scmm_v4_gated_quotes(
    S: float,
    q: int,
    drawdown: float,
    fill_imbalance: float,
    realized_vol: float,
    gamma: float,
    sigma: float,
    T: float,
    t: float,
    k: float,
    alpha_q: float,
    alpha_f: float,
    beta_q: float,
    beta_d: float,
    beta_sigma: float,
    eta_time: float,
    q_threshold: int,
    drawdown_threshold: float,
    imbalance_threshold: float,
    use_gating: bool = True,
    use_drawdown_term: bool = True,
    use_vol_term: bool = True,
    use_time_urgency: bool = True,
) -> tuple[float, float, float, float, float, float, float, float]:
    r_as, delta_as, _, _ = compute_quotes(
        S=S,
        q=q,
        gamma=gamma,
        sigma=sigma,
        T=T,
        t=t,
        k=k,
    )

    u_skew, u_spr = compute_scmm_v4_gated_corrections(
        q=q,
        drawdown=drawdown,
        fill_imbalance=fill_imbalance,
        realized_vol=realized_vol,
        t=t,
        T=T,
        alpha_q=alpha_q,
        alpha_f=alpha_f,
        beta_q=beta_q,
        beta_d=beta_d,
        beta_sigma=beta_sigma,
        eta_time=eta_time,
        q_threshold=q_threshold,
        drawdown_threshold=drawdown_threshold,
        imbalance_threshold=imbalance_threshold,
        use_gating=use_gating,
        use_drawdown_term=use_drawdown_term,
        use_vol_term=use_vol_term,
        use_time_urgency=use_time_urgency,
    )

    r_sc = r_as + u_skew
    delta_sc = max(delta_as + u_spr, 1e-6)

    bid_sc = r_sc - delta_sc
    ask_sc = r_sc + delta_sc

    return r_as, delta_as, r_sc, delta_sc, u_skew, u_spr, bid_sc, ask_sc