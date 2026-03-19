import numpy as np
import pandas as pd

from config.default_params import Params
from src.execution.fill_model import bid_ask_fill_probabilities, simulate_fill
from src.market.regime_switching import (
    next_regime,
    regime_parameters,
    step_price_regime_switching,
)
from src.policy.avellaneda_stoikov import compute_quotes
from src.portfolio.portfolio import Portfolio
from src.utils.logger import SimulationLogger


class ASRegimeBarrierSimulator:
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
        regime = 0
        running_max_wealth = self.portfolio.wealth(S)

        self.logger.log(
            {
                "step": 0,
                "time": 0.0,
                "mid_price": S,
                "regime": regime,
                "mu": np.nan,
                "sigma_used": np.nan,
                "reservation_price": np.nan,
                "half_spread": np.nan,
                "bid": np.nan,
                "ask": np.nan,
                "inventory": self.portfolio.inventory,
                "cash": self.portfolio.cash,
                "wealth": self.portfolio.wealth(S),
                "drawdown": 0.0,
                "barrier_hit": False,
                "survived": True,
                "lambda_bid": np.nan,
                "lambda_ask": np.nan,
                "prob_bid_fill": np.nan,
                "prob_ask_fill": np.nan,
                "bid_fill": False,
                "ask_fill": False,
            }
        )

        barrier_hit = False

        for step in range(1, self.num_steps + 1):
            t = (step - 1) * p.dt

            mu, sigma_used = regime_parameters(
                regime=regime,
                sigma_0=p.regime_sigma_0,
                sigma_1=p.regime_sigma_1,
                mu_0=p.regime_mu_0,
                mu_1=p.regime_mu_1,
            )

            r, delta, bid, ask = compute_quotes(
                S=S,
                q=self.portfolio.inventory,
                gamma=p.gamma,
                sigma=sigma_used,
                T=p.T,
                t=t,
                k=p.k,
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

            regime = next_regime(
                current_regime=regime,
                p00=p.regime_p00,
                p11=p.regime_p11,
                rng=self.rng,
            )

            mu_next, sigma_next = regime_parameters(
                regime=regime,
                sigma_0=p.regime_sigma_0,
                sigma_1=p.regime_sigma_1,
                mu_0=p.regime_mu_0,
                mu_1=p.regime_mu_1,
            )

            S = step_price_regime_switching(
                S=S,
                mu=mu_next,
                sigma=sigma_next,
                dt=p.dt,
                rng=self.rng,
            )

            wealth = self.portfolio.wealth(S)
            running_max_wealth = max(running_max_wealth, wealth)
            drawdown = running_max_wealth - wealth

            if drawdown >= p.barrier_drawdown_limit:
                barrier_hit = True
                wealth -= p.barrier_terminal_penalty

            self.logger.log(
                {
                    "step": step,
                    "time": step * p.dt,
                    "mid_price": S,
                    "regime": regime,
                    "mu": mu_next,
                    "sigma_used": sigma_used,
                    "reservation_price": r,
                    "half_spread": delta,
                    "bid": bid,
                    "ask": ask,
                    "inventory": self.portfolio.inventory,
                    "cash": self.portfolio.cash,
                    "wealth": wealth,
                    "drawdown": drawdown,
                    "barrier_hit": barrier_hit,
                    "survived": not barrier_hit,
                    "lambda_bid": lambda_bid,
                    "lambda_ask": lambda_ask,
                    "prob_bid_fill": prob_bid,
                    "prob_ask_fill": prob_ask,
                    "bid_fill": bid_fill,
                    "ask_fill": ask_fill,
                }
            )

            if barrier_hit:
                break

        return self.logger.to_dataframe()