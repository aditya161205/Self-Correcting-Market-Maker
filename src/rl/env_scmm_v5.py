from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config.default_params import Params
from src.execution.fill_model import bid_ask_fill_probabilities, simulate_fill
from src.market.regime_switching import next_regime, regime_parameters, step_price_regime_switching
from src.policy.avellaneda_stoikov import compute_quotes
from src.policy.scmm_rule import compute_scmm_v4_gated_corrections
from src.portfolio.portfolio import Portfolio


class SCMMV5Env(gym.Env):
    """
    SCMM-v5 = SCMM-v4 prior + RL residual correction.

    RL action is normalized in [-1, 1]^2 and is mapped to small residual
    skew/spread corrections added on top of SCMM-v4.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        params: Params | None = None,
        lambda_q: float = 0.05,
        lambda_d: float = 0.02,
        lambda_res: float = 0.02,
        residual_skew_max: float = 0.10,
        residual_spr_max: float = 0.05,
    ) -> None:
        super().__init__()

        self.params = params if params is not None else Params()

        self.lambda_q = float(lambda_q)
        self.lambda_d = float(lambda_d)
        self.lambda_res = float(lambda_res)

        self.residual_skew_max = float(residual_skew_max)
        self.residual_spr_max = float(residual_spr_max)

        # normalized symmetric action space
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        # observation:
        # [inventory, abs_inventory, time_remaining, drawdown, fill_imbalance,
        #  realized_vol, regime, half_spread_as, u_skew_v4, u_spr_v4]
        self.observation_space = spaces.Box(
            low=np.array([-10, 0, 0, 0, -10, 0, 0, 0, -10, 0], dtype=np.float32),
            high=np.array([10, 10, 1, 10, 10, 10, 1, 10, 10, 10], dtype=np.float32),
            dtype=np.float32,
        )

        self.rng: np.random.Generator | None = None
        self.portfolio: Portfolio | None = None

        self.S = 0.0
        self.current_step = 0
        self.running_max_wealth = 0.0
        self.regime = 0

        self.bid_fill_history = deque(maxlen=20)
        self.ask_fill_history = deque(maxlen=20)
        self.return_history = deque(maxlen=self.params.scmm_vol_window)

    def _time(self) -> float:
        return self.current_step * self.params.dt

    def _time_remaining(self) -> float:
        return max(self.params.T - self._time(), 0.0)

    def _fill_imbalance(self) -> float:
        return float(sum(self.bid_fill_history) - sum(self.ask_fill_history))

    def _realized_vol(self) -> float:
        if len(self.return_history) < 2:
            return 0.0
        return float(np.std(self.return_history) / self.params.scmm_vol_scale)

    def _drawdown(self) -> float:
        wealth = self.portfolio.wealth(self.S)
        return float(max(self.running_max_wealth - wealth, 0.0))

    def _as_quantities(self, sigma_used: float) -> tuple[float, float]:
        t = self._time()
        r_as, delta_as, _, _ = compute_quotes(
            S=self.S,
            q=self.portfolio.inventory,
            gamma=self.params.gamma,
            sigma=sigma_used,
            T=self.params.T,
            t=t,
            k=self.params.k,
        )
        return float(r_as), float(delta_as)

    def _v4_corrections(self, sigma_used: float) -> tuple[float, float, float, float]:
        t = self._time()
        r_as, delta_as = self._as_quantities(sigma_used=sigma_used)

        u_skew_v4, u_spr_v4 = compute_scmm_v4_gated_corrections(
            q=self.portfolio.inventory,
            drawdown=self._drawdown(),
            fill_imbalance=self._fill_imbalance(),
            realized_vol=self._realized_vol(),
            t=t,
            T=self.params.T,
            alpha_q=self.params.scmm_alpha_q,
            alpha_f=self.params.scmm_alpha_f,
            beta_q=self.params.scmm_beta_q,
            beta_d=self.params.scmm_beta_d,
            beta_sigma=self.params.scmm_beta_sigma,
            eta_time=self.params.scmm_eta_time,
            q_threshold=self.params.scmm_q_threshold,
            drawdown_threshold=self.params.scmm_drawdown_threshold,
            imbalance_threshold=self.params.scmm_imbalance_threshold,
            use_gating=self.params.scmm_use_gating,
            use_drawdown_term=self.params.scmm_use_drawdown_term,
            use_vol_term=self.params.scmm_use_vol_term,
            use_time_urgency=self.params.scmm_use_time_urgency,
        )
        return r_as, delta_as, float(u_skew_v4), float(u_spr_v4)

    def _obs(self, sigma_used: float) -> np.ndarray:
        r_as, delta_as, u_skew_v4, u_spr_v4 = self._v4_corrections(sigma_used=sigma_used)

        q = float(self.portfolio.inventory)
        obs = np.array(
            [
                q / 10.0,
                abs(q) / 10.0,
                self._time_remaining() / self.params.T,
                self._drawdown() / 10.0,
                self._fill_imbalance() / 20.0,
                self._realized_vol() / 5.0,
                float(self.regime),
                delta_as / 5.0,
                u_skew_v4 / 1.0,
                u_spr_v4 / 1.0,
            ],
            dtype=np.float32,
        )
        return obs

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)

        actual_seed = self.params.seed if seed is None else seed
        self.rng = np.random.default_rng(actual_seed)

        self.portfolio = Portfolio()
        self.S = float(self.params.S0)
        self.current_step = 0
        self.running_max_wealth = float(self.portfolio.wealth(self.S))
        self.regime = 0

        self.bid_fill_history.clear()
        self.ask_fill_history.clear()
        self.return_history.clear()

        _, sigma_used = regime_parameters(
            regime=self.regime,
            sigma_0=self.params.regime_sigma_0,
            sigma_1=self.params.regime_sigma_1,
            mu_0=self.params.regime_mu_0,
            mu_1=self.params.regime_mu_1,
        )

        return self._obs(sigma_used=sigma_used), {}

    def step(self, action: np.ndarray):
        p = self.params

        mu, sigma_used = regime_parameters(
            regime=self.regime,
            sigma_0=p.regime_sigma_0,
            sigma_1=p.regime_sigma_1,
            mu_0=p.regime_mu_0,
            mu_1=p.regime_mu_1,
        )

        wealth_before = float(self.portfolio.wealth(self.S))
        self.running_max_wealth = max(self.running_max_wealth, wealth_before)

        r_as, delta_as, u_skew_v4, u_spr_v4 = self._v4_corrections(sigma_used=sigma_used)

        # residual RL action
        a0 = float(np.clip(action[0], -1.0, 1.0))
        a1 = float(np.clip(action[1], -1.0, 1.0))

        du_skew = a0 * self.residual_skew_max
        du_spr = a1 * self.residual_spr_max

        u_skew = u_skew_v4 + du_skew
        u_spr = u_spr_v4 + du_spr

        r_sc = r_as + u_skew
        delta_sc = max(delta_as + u_spr, 1e-6)

        bid = r_sc - delta_sc
        ask = r_sc + delta_sc

        lambda_bid, lambda_ask, prob_bid, prob_ask = bid_ask_fill_probabilities(
            S=self.S,
            bid=bid,
            ask=ask,
            A=p.A,
            k=p.k,
            dt=p.dt,
        )

        bid_fill = simulate_fill(prob_bid, self.rng)
        ask_fill = simulate_fill(prob_ask, self.rng)

        if p.max_inventory is not None:
            if bid_fill and self.portfolio.inventory >= p.max_inventory:
                bid_fill = False
            if ask_fill and self.portfolio.inventory <= -p.max_inventory:
                ask_fill = False

        if bid_fill:
            self.portfolio.update_on_bid_fill(bid)
        if ask_fill:
            self.portfolio.update_on_ask_fill(ask)

        self.bid_fill_history.append(1 if bid_fill else 0)
        self.ask_fill_history.append(1 if ask_fill else 0)

        old_S = self.S

        self.regime = next_regime(
            current_regime=self.regime,
            p00=p.regime_p00,
            p11=p.regime_p11,
            rng=self.rng,
        )

        mu_next, sigma_next = regime_parameters(
            regime=self.regime,
            sigma_0=p.regime_sigma_0,
            sigma_1=p.regime_sigma_1,
            mu_0=p.regime_mu_0,
            mu_1=p.regime_mu_1,
        )

        self.S = step_price_regime_switching(
            S=self.S,
            mu=mu_next,
            sigma=sigma_next,
            dt=p.dt,
            rng=self.rng,
        )

        self.return_history.append(self.S - old_S)

        wealth_after = float(self.portfolio.wealth(self.S))
        self.running_max_wealth = max(self.running_max_wealth, wealth_after)
        drawdown_after = float(max(self.running_max_wealth - wealth_after, 0.0))

        delta_wealth = wealth_after - wealth_before
        inventory_penalty = self.lambda_q * float(self.portfolio.inventory ** 2)
        drawdown_penalty = self.lambda_d * drawdown_after
        residual_penalty = self.lambda_res * (du_skew ** 2 + du_spr ** 2)

        reward = delta_wealth - inventory_penalty - drawdown_penalty - residual_penalty

        self.current_step += 1

        barrier_hit = drawdown_after >= p.barrier_drawdown_limit
        terminated = barrier_hit or (self.current_step >= int(round(p.T / p.dt)))
        truncated = False

        obs = self._obs(sigma_used=sigma_next)
        info = {
            "wealth_after": wealth_after,
            "inventory": int(self.portfolio.inventory),
            "drawdown_after": drawdown_after,
            "barrier_hit": bool(barrier_hit),
            "survived": not barrier_hit,
            "u_skew_v4": u_skew_v4,
            "u_spr_v4": u_spr_v4,
            "du_skew_rl": du_skew,
            "du_spr_rl": du_spr,
            "u_skew_total": u_skew,
            "u_spr_total": u_spr,
            "regime": int(self.regime),
            "bid_fill": bool(bid_fill),
            "ask_fill": bool(ask_fill),
            "bid": float(bid),
            "ask": float(ask),
            "mid_price": float(self.S),
            "reward": float(reward),
            "time": float(self._time()),
        }

        return obs, float(reward), terminated, truncated, info

    def render(self):
        return None

    def close(self):
        return None