from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config.default_params import Params
from src.execution.fill_model import bid_ask_fill_probabilities, simulate_fill
from src.market.price_process import step_price
from src.policy.avellaneda_stoikov import compute_quotes
from src.portfolio.portfolio import Portfolio


class SCMMEnv(gym.Env):
    """
    Gymnasium environment for risk-aware self-correcting market making.

    The agent outputs corrections on top of Avellaneda–Stoikov:
        action = [u_skew, u_spr]

    Improved observation:
        [
            inventory_norm,
            abs_inventory_norm,
            time_remaining_norm,
            drawdown_norm,
            fill_imbalance_norm,
            half_spread_as_norm,
            reservation_offset_norm,
        ]

    Reward:
        delta_wealth
        - lambda_q * inventory^2
        - lambda_d * drawdown
        - lambda_a * (u_skew^2 + u_spr^2)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        params: Params | None = None,
        lambda_q: float = 0.05,
        lambda_d: float = 0.02,
        lambda_a: float = 0.01,
        max_u_skew: float = 0.5,
        max_u_spr: float = 0.5,
        fill_imbalance_window: int = 20,
        inventory_scale: float = 10.0,
        drawdown_scale: float = 10.0,
        spread_scale: float = 5.0,
        reservation_offset_scale: float = 5.0,
        fill_imbalance_scale: float = 20.0,
    ) -> None:
        super().__init__()

        self.params = params if params is not None else Params()

        self.lambda_q = float(lambda_q)
        self.lambda_d = float(lambda_d)
        self.lambda_a = float(lambda_a)

        self.max_u_skew = float(max_u_skew)
        self.max_u_spr = float(max_u_spr)
        self.fill_imbalance_window = int(fill_imbalance_window)

        # normalization scales
        self.inventory_scale = float(inventory_scale)
        self.drawdown_scale = float(drawdown_scale)
        self.spread_scale = float(spread_scale)
        self.reservation_offset_scale = float(reservation_offset_scale)
        self.fill_imbalance_scale = float(fill_imbalance_scale)

        # Action = [u_skew, u_spr]
        self.action_space = spaces.Box(
            low=np.array([-self.max_u_skew, 0.0], dtype=np.float32),
            high=np.array([self.max_u_skew, self.max_u_spr], dtype=np.float32),
            dtype=np.float32,
        )

        # Normalized observation, mostly around [-1, 1]-ish, but keep wide finite bounds
        self.observation_space = spaces.Box(
            low=np.array([-10.0, 0.0, 0.0, 0.0, -10.0, 0.0, -10.0], dtype=np.float32),
            high=np.array([10.0, 10.0, 1.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.rng: np.random.Generator | None = None
        self.portfolio: Portfolio | None = None

        self.S: float = 0.0
        self.current_step: int = 0
        self.running_max_wealth: float = 0.0

        self.bid_fill_history: deque[int] = deque(maxlen=self.fill_imbalance_window)
        self.ask_fill_history: deque[int] = deque(maxlen=self.fill_imbalance_window)

    def _get_time(self) -> float:
        return self.current_step * self.params.dt

    def _get_time_remaining(self) -> float:
        return max(self.params.T - self._get_time(), 0.0)

    def _get_recent_fill_imbalance(self) -> float:
        return float(sum(self.bid_fill_history) - sum(self.ask_fill_history))

    def _get_drawdown(self) -> float:
        wealth = self.portfolio.wealth(self.S)
        return float(max(self.running_max_wealth - wealth, 0.0))

    def _get_as_quantities(self) -> tuple[float, float]:
        t = self._get_time()
        r_as, delta_as, _, _ = compute_quotes(
            S=self.S,
            q=self.portfolio.inventory,
            gamma=self.params.gamma,
            sigma=self.params.sigma,
            T=self.params.T,
            t=t,
            k=self.params.k,
        )
        return float(r_as), float(delta_as)

    def _normalize_obs(
        self,
        inventory: float,
        time_remaining: float,
        drawdown: float,
        fill_imbalance: float,
        half_spread_as: float,
        reservation_offset: float,
    ) -> np.ndarray:
        inventory_norm = inventory / self.inventory_scale
        abs_inventory_norm = abs(inventory) / self.inventory_scale
        time_remaining_norm = time_remaining / self.params.T if self.params.T > 0 else 0.0
        drawdown_norm = drawdown / self.drawdown_scale
        fill_imbalance_norm = fill_imbalance / self.fill_imbalance_scale
        half_spread_as_norm = half_spread_as / self.spread_scale
        reservation_offset_norm = reservation_offset / self.reservation_offset_scale

        return np.array(
            [
                inventory_norm,
                abs_inventory_norm,
                time_remaining_norm,
                drawdown_norm,
                fill_imbalance_norm,
                half_spread_as_norm,
                reservation_offset_norm,
            ],
            dtype=np.float32,
        )

    def _get_obs(self) -> np.ndarray:
        q = float(self.portfolio.inventory)
        time_remaining = self._get_time_remaining()
        drawdown = self._get_drawdown()
        fill_imbalance = self._get_recent_fill_imbalance()
        r_as, delta_as = self._get_as_quantities()
        reservation_offset = r_as - self.S

        return self._normalize_obs(
            inventory=q,
            time_remaining=time_remaining,
            drawdown=drawdown,
            fill_imbalance=fill_imbalance,
            half_spread_as=delta_as,
            reservation_offset=reservation_offset,
        )

    def _get_info(self) -> dict[str, Any]:
        r_as, delta_as = self._get_as_quantities()

        return {
            "mid_price": float(self.S),
            "wealth": float(self.portfolio.wealth(self.S)),
            "inventory": int(self.portfolio.inventory),
            "cash": float(self.portfolio.cash),
            "drawdown": float(self._get_drawdown()),
            "time": float(self._get_time()),
            "time_remaining": float(self._get_time_remaining()),
            "fill_imbalance": float(self._get_recent_fill_imbalance()),
            "reservation_price_as": float(r_as),
            "half_spread_as": float(delta_as),
            "reservation_offset_as": float(r_as - self.S),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        actual_seed = self.params.seed if seed is None else seed
        self.rng = np.random.default_rng(actual_seed)

        self.portfolio = Portfolio()
        self.S = float(self.params.S0)
        self.current_step = 0
        self.running_max_wealth = float(self.portfolio.wealth(self.S))

        self.bid_fill_history.clear()
        self.ask_fill_history.clear()

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.rng is None or self.portfolio is None:
            raise RuntimeError("Environment must be reset before calling step().")

        p = self.params
        t = self._get_time()

        raw_u_skew = float(action[0])
        raw_u_spr = float(action[1])

        u_skew = float(np.clip(raw_u_skew, -self.max_u_skew, self.max_u_skew))
        u_spr = float(np.clip(raw_u_spr, 0.0, self.max_u_spr))

        wealth_before = float(self.portfolio.wealth(self.S))
        self.running_max_wealth = max(self.running_max_wealth, wealth_before)
        drawdown_before = float(max(self.running_max_wealth - wealth_before, 0.0))

        r_as, delta_as, _, _ = compute_quotes(
            S=self.S,
            q=self.portfolio.inventory,
            gamma=p.gamma,
            sigma=p.sigma,
            T=p.T,
            t=t,
            k=p.k,
        )

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

        self.S = step_price(S=self.S, sigma=p.sigma, dt=p.dt, rng=self.rng)

        wealth_after = float(self.portfolio.wealth(self.S))
        delta_wealth = wealth_after - wealth_before

        self.running_max_wealth = max(self.running_max_wealth, wealth_after)
        drawdown_after = float(max(self.running_max_wealth - wealth_after, 0.0))

        inventory_penalty = self.lambda_q * float(self.portfolio.inventory ** 2)
        drawdown_penalty = self.lambda_d * drawdown_after
        action_penalty = self.lambda_a * (u_skew ** 2 + u_spr ** 2)

        reward = delta_wealth - inventory_penalty - drawdown_penalty - action_penalty

        self.current_step += 1

        terminated = self.current_step >= int(round(p.T / p.dt))
        truncated = False

        obs = self._get_obs()
        info = {
            "mid_price": float(self.S),
            "wealth_before": float(wealth_before),
            "wealth_after": float(wealth_after),
            "delta_wealth": float(delta_wealth),
            "inventory": int(self.portfolio.inventory),
            "cash": float(self.portfolio.cash),
            "drawdown_before": float(drawdown_before),
            "drawdown_after": float(drawdown_after),
            "inventory_penalty": float(inventory_penalty),
            "drawdown_penalty": float(drawdown_penalty),
            "action_penalty": float(action_penalty),
            "reward": float(reward),
            "time": float(self._get_time()),
            "time_remaining": float(self._get_time_remaining()),
            "fill_imbalance": float(self._get_recent_fill_imbalance()),
            "reservation_price_as": float(r_as),
            "half_spread_as": float(delta_as),
            "reservation_offset_as": float(r_as - self.S),
            "reservation_price_sc": float(r_sc),
            "half_spread_sc": float(delta_sc),
            "u_skew": float(u_skew),
            "u_spr": float(u_spr),
            "bid": float(bid),
            "ask": float(ask),
            "lambda_bid": float(lambda_bid),
            "lambda_ask": float(lambda_ask),
            "prob_bid_fill": float(prob_bid),
            "prob_ask_fill": float(prob_ask),
            "bid_fill": bool(bid_fill),
            "ask_fill": bool(ask_fill),
        }

        return obs, float(reward), terminated, truncated, info

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None