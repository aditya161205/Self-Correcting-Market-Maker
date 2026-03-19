from __future__ import annotations

from dataclasses import replace

import pandas as pd

from config.default_params import Params
from src.metrics.performance import compute_all_metrics
from src.simulator.as_regime_barrier_simulator import ASRegimeBarrierSimulator
from src.simulator.scmm_v4_gated_regime_barrier_simulator import SCMMV4GatedRegimeBarrierSimulator
from src.utils.helpers import ensure_dir


def run_model(simulator_cls, params: Params, n_runs: int = 50, base_seed: int = 42) -> dict:
    rows = []

    for i in range(n_runs):
        seed = base_seed + i
        run_params = replace(params, seed=seed)

        sim = simulator_cls(run_params)
        df = sim.run()

        metrics = compute_all_metrics(df)
        metrics["barrier_hit"] = bool(df["barrier_hit"].iloc[-1])
        metrics["survived"] = bool(df["survived"].iloc[-1])
        metrics["steps_survived"] = int(len(df) - 1)

        rows.append(metrics)

    df_runs = pd.DataFrame(rows)

    return {
        "n_runs": n_runs,
        "mean_final_wealth": float(df_runs["final_wealth"].mean()),
        "std_final_wealth": float(df_runs["final_wealth"].std()),
        "mean_sharpe_like": float(df_runs["sharpe_like"].mean()),
        "final_wealth_p05": float(df_runs["final_wealth"].quantile(0.05)),
        "mean_abs_inventory": float(df_runs["mean_abs_inventory"].mean()),
        "mean_inventory_std": float(df_runs["inventory_std"].mean()),
        "mean_max_drawdown": float(df_runs["max_drawdown"].mean()),
        "mean_mean_drawdown": float(df_runs["mean_drawdown"].mean()),
        "mean_fill_imbalance": float(df_runs["fill_imbalance"].mean()),
        "mean_total_fills": float(df_runs["total_fills"].mean()),
        "barrier_hit_rate": float(df_runs["barrier_hit"].mean()),
        "survival_rate": float(df_runs["survived"].mean()),
        "mean_steps_survived": float(df_runs["steps_survived"].mean()),
    }


def main() -> None:
    output_dir = ensure_dir("outputs")
    results_dir = ensure_dir(output_dir / "results")

    base_params = Params()

    experiments = []

    # 1. AS baseline
    experiments.append(("AS_regime_barrier", ASRegimeBarrierSimulator, base_params))

    # 2. Full winning SCMM-v4
    experiments.append(("SCMM_v4_full", SCMMV4GatedRegimeBarrierSimulator, base_params))

    # 3. No gating
    experiments.append((
        "SCMM_v4_no_gating",
        SCMMV4GatedRegimeBarrierSimulator,
        replace(base_params, scmm_use_gating=False),
    ))

    # 4. No drawdown term
    experiments.append((
        "SCMM_v4_no_drawdown",
        SCMMV4GatedRegimeBarrierSimulator,
        replace(base_params, scmm_use_drawdown_term=False),
    ))

    # 5. No volatility term
    experiments.append((
        "SCMM_v4_no_vol",
        SCMMV4GatedRegimeBarrierSimulator,
        replace(base_params, scmm_use_vol_term=False),
    ))

    # 6. No time urgency
    experiments.append((
        "SCMM_v4_no_time_urgency",
        SCMMV4GatedRegimeBarrierSimulator,
        replace(base_params, scmm_use_time_urgency=False),
    ))

    rows = []
    for name, simulator_cls, params in experiments:
        summary = run_model(simulator_cls, params, n_runs=50, base_seed=42)
        summary["model"] = name
        rows.append(summary)
        print(
            f"{name}: wealth={summary['mean_final_wealth']:.4f}, "
            f"sharpe={summary['mean_sharpe_like']:.6f}, "
            f"abs_inv={summary['mean_abs_inventory']:.6f}, "
            f"max_dd={summary['mean_max_drawdown']:.6f}, "
            f"barrier_hit={summary['barrier_hit_rate']:.2%}"
        )

    df = pd.DataFrame(rows)

    cols = [
        "model",
        "n_runs",
        "mean_final_wealth",
        "std_final_wealth",
        "mean_sharpe_like",
        "final_wealth_p05",
        "mean_abs_inventory",
        "mean_inventory_std",
        "mean_max_drawdown",
        "mean_mean_drawdown",
        "mean_fill_imbalance",
        "mean_total_fills",
        "barrier_hit_rate",
        "survival_rate",
        "mean_steps_survived",
    ]
    df = df[cols]

    df.to_csv(results_dir / "scmm_v4_ablation_suite_summary.csv", index=False)

    print("\nAblation suite summary:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()