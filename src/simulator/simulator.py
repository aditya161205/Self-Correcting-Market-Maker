import numpy as np
import pandas as pd

from config.default_params import Params
from src.market.price_process import step_price
from src.policy.avellaneda_stoikov import compute_quotes
from src.execution.fill_model import bid_ask_fill_probabilities, simulate_fill
from src.portfolio.portfolio import Portfolio
from src.utils.logger import SimulationLogger


class AvellanedaStoikovSimulator:
    def __init__(self, params: Params):
        self.params = params
        self.rng = np.random.default_rng(params.seed)
        self.portfolio = Portfolio()
        self.logger = SimulationLogger()

        self.num_steps = int(round(params.T / params.dt))

    def _inventory_cap_blocks(self, bid_fill: bool, ask_fill: bool) -> tuple[bool, bool]:
        """
        Optional safety cap on inventory.
        If max_inventory is None, do nothing.
        """
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

        # log initial state at t=0
        initial_wealth = self.portfolio.wealth(S)
        self.logger.log(
            {
                "step": 0,
                "time": 0.0,
                "mid_price": S,
                "reservation_price": np.nan,
                "half_spread": np.nan,
                "bid": np.nan,
                "ask": np.nan,
                "inventory": self.portfolio.inventory,
                "cash": self.portfolio.cash,
                "wealth": initial_wealth,
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

            # 1. Compute AS quotes using current state
            r, delta, bid, ask = compute_quotes(
                S=S,
                q=self.portfolio.inventory,
                gamma=p.gamma,
                sigma=p.sigma,
                T=p.T,
                t=t,
                k=p.k,
            )

            # 2. Compute fill probabilities
            lambda_bid, lambda_ask, prob_bid, prob_ask = bid_ask_fill_probabilities(
                S=S,
                bid=bid,
                ask=ask,
                A=p.A,
                k=p.k,
                dt=p.dt,
            )

            # 3. Simulate independent bid/ask fills
            bid_fill = simulate_fill(prob_bid, self.rng)
            ask_fill = simulate_fill(prob_ask, self.rng)

            # 4. Optional inventory cap
            bid_fill, ask_fill = self._inventory_cap_blocks(bid_fill, ask_fill)

            # 5. Update portfolio from fills
            if bid_fill:
                self.portfolio.update_on_bid_fill(bid)

            if ask_fill:
                self.portfolio.update_on_ask_fill(ask)

            # 6. Evolve price to next step
            S = step_price(S=S, sigma=p.sigma, dt=p.dt, rng=self.rng)

            # 7. Mark wealth
            wealth = self.portfolio.wealth(S)

            # 8. Log state
            self.logger.log(
                {
                    "step": step,
                    "time": step * p.dt,
                    "mid_price": S,
                    "reservation_price": r,
                    "half_spread": delta,
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