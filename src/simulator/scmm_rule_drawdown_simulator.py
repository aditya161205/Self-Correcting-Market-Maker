import numpy as np
import pandas as pd

from config.default_params import Params
from src.execution.fill_model import bid_ask_fill_probabilities, simulate_fill
from src.market.price_process import step_price
from src.policy.scmm_rule import compute_scmm_quotes_inventory_drawdown
from src.portfolio.portfolio import Portfolio
from src.utils.logger import SimulationLogger


class SCMRuleDrawdownSimulator:
    def __init__(self, params: Params):
        self.params = params
        self.rng = np.random.default_rng(params.seed)
        self.portfolio = Portfolio()
        self.logger = SimulationLogger()
        self.num_steps = int(round(params.T / params.dt))

    def _inventory_cap_blocks(self, bid_fill: bool, ask_fill: bool) -> tuple[bool, bool]:
        max_inv = self.params.max_inventory
        q = self.portfolio.inventory

        if max_inv is None:
            return bid_fill, ask_fill

        if bid_fill and q >= max_inv:
            bid_fill = False
        if ask_fill and q <= -max_inv:
            ask_fill = False

        return bid_fill, ask_fill

    def run(self) -> pd.DataFrame:
        p = self.params
        S = p.S0

        running_max_wealth = self.portfolio.wealth(S)

        self.logger.log(
            {
                "step": 0,
                "time": 0.0,
                "mid_price": S,
                "reservation_price_as": np.nan,
                "half_spread_as": np.nan,
                "reservation_price_sc": np.nan,
                "half_spread_sc": np.nan,
                "drawdown": 0.0,
                "u_skew": np.nan,
                "u_spr": np.nan,
                "bid": np.nan,
                "ask": np.nan,
                "inventory": self.portfolio.inventory,
                "cash": self.portfolio.cash,
                "wealth": self.portfolio.wealth(S),
                "lambda_bid": np.nan,
                "lambda_ask": np.nan,
                "prob_bid_fill": np.nan,
                "prob_ask_fill": np.nan,
                "bid_fill": False,
                "ask_fill": False,
            }
        )

        for step in range(1, self.num_steps + 1):
            t = (step - 1) * p.dt

            current_wealth = self.portfolio.wealth(S)
            running_max_wealth = max(running_max_wealth, current_wealth)
            drawdown = running_max_wealth - current_wealth

            (
                r_as,
                delta_as,
                r_sc,
                delta_sc,
                u_skew,
                u_spr,
                bid,
                ask,
            ) = compute_scmm_quotes_inventory_drawdown(
                S=S,
                q=self.portfolio.inventory,
                drawdown=drawdown,
                gamma=p.gamma,
                sigma=p.sigma,
                T=p.T,
                t=t,
                k=p.k,
                alpha_q=p.scmm_alpha_q,
                beta_q=p.scmm_beta_q,
                beta_d=p.scmm_beta_d,
            )

            lambda_bid, lambda_ask, prob_bid, prob_ask = bid_ask_fill_probabilities(
                S=S,
                bid=bid,
                ask=ask,
                A=p.A,
                k=p.k,
                dt=p.dt,
            )

            bid_fill = simulate_fill(prob_bid, self.rng)
            ask_fill = simulate_fill(prob_ask, self.rng)

            bid_fill, ask_fill = self._inventory_cap_blocks(bid_fill, ask_fill)

            if bid_fill:
                self.portfolio.update_on_bid_fill(bid)

            if ask_fill:
                self.portfolio.update_on_ask_fill(ask)

            S = step_price(S=S, sigma=p.sigma, dt=p.dt, rng=self.rng)
            wealth = self.portfolio.wealth(S)

            self.logger.log(
                {
                    "step": step,
                    "time": step * p.dt,
                    "mid_price": S,
                    "reservation_price_as": r_as,
                    "half_spread_as": delta_as,
                    "reservation_price_sc": r_sc,
                    "half_spread_sc": delta_sc,
                    "drawdown": drawdown,
                    "u_skew": u_skew,
                    "u_spr": u_spr,
                    "bid": bid,
                    "ask": ask,
                    "inventory": self.portfolio.inventory,
                    "cash": self.portfolio.cash,
                    "wealth": wealth,
                    "lambda_bid": lambda_bid,
                    "lambda_ask": lambda_ask,
                    "prob_bid_fill": prob_bid,
                    "prob_ask_fill": prob_ask,
                    "bid_fill": bid_fill,
                    "ask_fill": ask_fill,
                }
            )

        return self.logger.to_dataframe()