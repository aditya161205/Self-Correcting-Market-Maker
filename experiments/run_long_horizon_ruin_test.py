from __future__ import annotations

from dataclasses import replace

import pandas as pd

from config.default_params import Params
from src.metrics.performance import compute_all_metrics
from src.simulator.as_regime_barrier_simulator import ASRegimeBarrierSimulator
from src.simulator.scmm_v4_gated_regime_barrier_simulator import SCMMV4GatedRegimeBarrierSimulator
from src.utils.helpers import ensure_dir


def run_model(simulator_cls, params: Params, n_runs: int = 300, base_seed: int = 42) -> pd.DataFrame:
    rows = []

    for i in range(n_runs):
        seed = base_seed + i
        run_params = replace(params, seed=seed)

        sim = simulator_cls(run_params)
        df = sim.run()

        metrics = compute_all_metrics(df)
        metrics["barrier_hit"] = int(bool(df["barrier_hit"].iloc[-1]))
        metrics["survived"] = int(bool(df["survived"].iloc[-1]))
        metrics["steps_survived"] = int(len(df) - 1)
        rows.append(metrics)

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, label: str, T: float) -> dict:
    return {
        "model": label,
        "T": T,
        "n_runs": int(len(df)),
        "mean_final_wealth": float(df["final_wealth"].mean()),
        "mean_sharpe_like": float(df["sharpe_like"].mean()),
        "mean_abs_inventory": float(df["mean_abs_inventory"].mean()),
        "mean_max_drawdown": float(df["max_drawdown"].mean()),
        "barrier_hit_rate": float(df["barrier_hit"].mean()),
        "survival_rate": float(df["survived"].mean()),
        "mean_steps_survived": float(df["steps_survived"].mean()),
    }


def main() -> None:
    output_dir = ensure_dir("outputs")
    results_dir = ensure_dir(output_dir / "results")

    horizons = [5.0, 10.0, 20.0, 40.0]
    rows = []

    for T in horizons:
        params = replace(Params(), T=T)

        print(f"Running long-horizon ruin test for T={T}...")

        df_as = run_model(ASRegimeBarrierSimulator, params, n_runs=300, base_seed=42)
        df_scmm = run_model(SCMMV4GatedRegimeBarrierSimulator, params, n_runs=300, base_seed=42)

        rows.append(summarize(df_as, "AS_regime_barrier", T))
        rows.append(summarize(df_scmm, "SCMM_v4_gated_regime_barrier", T))

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(results_dir / "long_horizon_ruin_test_summary.csv", index=False)

    print("\nLong-horizon ruin test summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()