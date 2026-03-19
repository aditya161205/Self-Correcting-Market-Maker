from __future__ import annotations

import itertools
from dataclasses import replace

import pandas as pd

from config.default_params import Params
from src.metrics.performance import compute_all_metrics
from src.simulator.scmm_v4_gated_simulator import SCMMV4GatedSimulator
from src.utils.helpers import ensure_dir


def run_one(params: Params, seed: int) -> dict:
    run_params = replace(params, seed=seed)
    sim = SCMMV4GatedSimulator(run_params)
    df = sim.run()
    metrics = compute_all_metrics(df)
    metrics["seed"] = seed
    return metrics


def evaluate_config(params: Params, n_runs: int = 30, base_seed: int = 42) -> tuple[pd.DataFrame, dict]:
    rows = []
    for i in range(n_runs):
        seed = base_seed + i
        rows.append(run_one(params, seed))

    df_runs = pd.DataFrame(rows)

    aggregate = {
        "n_runs": n_runs,
        "mean_final_wealth": float(df_runs["final_wealth"].mean()),
        "std_final_wealth": float(df_runs["final_wealth"].std()),
        "mean_mean_step_pnl": float(df_runs["mean_step_pnl"].mean()),
        "mean_std_step_pnl": float(df_runs["std_step_pnl"].mean()),
        "mean_sharpe_like": float(df_runs["sharpe_like"].mean()),
        "min_final_wealth": float(df_runs["final_wealth"].min()),
        "max_final_wealth": float(df_runs["final_wealth"].max()),
        "final_wealth_p05": float(df_runs["final_wealth"].quantile(0.05)),
        "mean_final_inventory": float(df_runs["final_inventory"].mean()),
        "mean_abs_inventory": float(df_runs["mean_abs_inventory"].mean()),
        "mean_inventory_std": float(df_runs["inventory_std"].mean()),
        "mean_bid_fills": float(df_runs["num_bid_fills"].mean()),
        "mean_ask_fills": float(df_runs["num_ask_fills"].mean()),
        "mean_total_fills": float(df_runs["total_fills"].mean()),
        "mean_fill_imbalance": float(df_runs["fill_imbalance"].mean()),
        "mean_max_drawdown": float(df_runs["max_drawdown"].mean()),
        "mean_mean_drawdown": float(df_runs["mean_drawdown"].mean()),
        "max_of_max_drawdown": float(df_runs["max_drawdown"].max()),
        "mean_bid_change_std": float(df_runs["bid_change_std"].mean()),
        "mean_ask_change_std": float(df_runs["ask_change_std"].mean()),
        "mean_spread_change_std": float(df_runs["spread_change_std"].mean()),
    }
    return df_runs, aggregate


def main() -> None:
    # AS baseline targets for acceptance
    as_mean_sharpe_like = 0.14876625335206728
    as_mean_max_drawdown = 2.7125418710178413
    as_mean_abs_inventory = 0.5042315369261476

    alpha_q_grid = [0.10, 0.15, 0.20]
    alpha_f_grid = [0.00, 0.02, 0.03]
    beta_q_grid = [0.02, 0.05]
    beta_d_grid = [0.00, 0.01, 0.015]

    base_params = Params()

    output_dir = ensure_dir("outputs")
    results_dir = ensure_dir(output_dir / "results")

    summary_rows = []

    combos = list(itertools.product(alpha_q_grid, alpha_f_grid, beta_q_grid, beta_d_grid))
    total = len(combos)

    for idx, (alpha_q, alpha_f, beta_q, beta_d) in enumerate(combos, start=1):
        params = replace(
            base_params,
            scmm_alpha_q=alpha_q,
            scmm_alpha_f=alpha_f,
            scmm_beta_q=beta_q,
            scmm_beta_d=beta_d,
        )

        _, agg = evaluate_config(params=params, n_runs=30, base_seed=42)

        agg["scmm_alpha_q"] = alpha_q
        agg["scmm_alpha_f"] = alpha_f
        agg["scmm_beta_q"] = beta_q
        agg["scmm_beta_d"] = beta_d

        # acceptance flags against AS
        agg["beats_as_sharpe"] = agg["mean_sharpe_like"] > as_mean_sharpe_like
        agg["beats_as_drawdown"] = agg["mean_max_drawdown"] < as_mean_max_drawdown
        agg["beats_as_inventory"] = agg["mean_abs_inventory"] < as_mean_abs_inventory
        agg["accept_core"] = (
            agg["beats_as_sharpe"]
            and agg["beats_as_drawdown"]
            and agg["beats_as_inventory"]
        )

        summary_rows.append(agg)

        print(
            f"[{idx}/{total}] "
            f"alpha_q={alpha_q:.2f}, alpha_f={alpha_f:.2f}, "
            f"beta_q={beta_q:.3f}, beta_d={beta_d:.3f} | "
            f"wealth={agg['mean_final_wealth']:.4f}, "
            f"sharpe={agg['mean_sharpe_like']:.6f}, "
            f"abs_inv={agg['mean_abs_inventory']:.6f}, "
            f"max_dd={agg['mean_max_drawdown']:.6f}, "
            f"accept={agg['accept_core']}"
        )

    df_summary = pd.DataFrame(summary_rows)

    # sort by acceptance first, then sharpe, then wealth
    df_sorted = df_summary.sort_values(
        by=["accept_core", "mean_sharpe_like", "mean_final_wealth"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    df_summary.to_csv(results_dir / "sweep_scmm_v4_gated_summary_unsorted.csv", index=False)
    df_sorted.to_csv(results_dir / "sweep_scmm_v4_gated_summary_sorted.csv", index=False)

    accepted = df_sorted[df_sorted["accept_core"]].copy()
    accepted.to_csv(results_dir / "sweep_scmm_v4_gated_accepted_only.csv", index=False)

    print("\nTop 10 configurations:")
    print(
        df_sorted[
            [
                "scmm_alpha_q",
                "scmm_alpha_f",
                "scmm_beta_q",
                "scmm_beta_d",
                "mean_final_wealth",
                "mean_sharpe_like",
                "mean_abs_inventory",
                "mean_max_drawdown",
                "accept_core",
            ]
        ].head(10).to_string(index=False)
    )

    if not accepted.empty:
        best = accepted.sort_values(by=["mean_final_wealth"], ascending=False).iloc[0]
        print("\nBest accepted configuration:")
        print(best.to_string())
    else:
        print("\nNo configuration beat AS on all three core criteria.")
        best_risk = df_sorted.iloc[0]
        print("\nBest near-miss configuration:")
        print(best_risk.to_string())


if __name__ == "__main__":
    main()