from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, ttest_ind

from config.default_params import Params
from src.metrics.performance import compute_all_metrics
from src.simulator.as_regime_barrier_simulator import ASRegimeBarrierSimulator
from src.simulator.scmm_v4_gated_regime_barrier_simulator import SCMMV4GatedRegimeBarrierSimulator
from src.utils.helpers import ensure_dir


def run_model_per_run(simulator_cls, params: Params, n_runs: int = 1000, base_seed: int = 42) -> pd.DataFrame:
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


def welch_pvalue(x: pd.Series, y: pd.Series) -> float:
    return float(ttest_ind(x, y, equal_var=False, nan_policy="omit").pvalue)


def mannwhitney_pvalue(x: pd.Series, y: pd.Series) -> float:
    return float(mannwhitneyu(x, y, alternative="two-sided").pvalue)


def barrier_fisher_pvalue(as_hits: pd.Series, scmm_hits: pd.Series) -> float:
    as_hit = int(as_hits.sum())
    as_survive = int(len(as_hits) - as_hit)
    scmm_hit = int(scmm_hits.sum())
    scmm_survive = int(len(scmm_hits) - scmm_hit)

    table = np.array([[as_hit, as_survive], [scmm_hit, scmm_survive]])
    _, p = fisher_exact(table, alternative="two-sided")
    return float(p)


def summarize(df: pd.DataFrame, label: str) -> dict:
    return {
        "model": label,
        "n_runs": int(len(df)),
        "mean_final_wealth": float(df["final_wealth"].mean()),
        "std_final_wealth": float(df["final_wealth"].std()),
        "mean_sharpe_like": float(df["sharpe_like"].mean()),
        "final_wealth_p05": float(df["final_wealth"].quantile(0.05)),
        "mean_abs_inventory": float(df["mean_abs_inventory"].mean()),
        "mean_inventory_std": float(df["inventory_std"].mean()),
        "mean_max_drawdown": float(df["max_drawdown"].mean()),
        "mean_mean_drawdown": float(df["mean_drawdown"].mean()),
        "mean_fill_imbalance": float(df["fill_imbalance"].mean()),
        "mean_total_fills": float(df["total_fills"].mean()),
        "barrier_hit_rate": float(df["barrier_hit"].mean()),
        "survival_rate": float(df["survived"].mean()),
        "mean_steps_survived": float(df["steps_survived"].mean()),
    }


def main() -> None:
    output_dir = ensure_dir("outputs")
    results_dir = ensure_dir(output_dir / "results")

    params = Params()
    n_runs = 1000

    print(f"Running large-sample significance comparison with {n_runs} runs/model...")

    df_as = run_model_per_run(ASRegimeBarrierSimulator, params, n_runs=n_runs, base_seed=42)
    df_scmm = run_model_per_run(SCMMV4GatedRegimeBarrierSimulator, params, n_runs=n_runs, base_seed=42)

    df_as.to_csv(results_dir / "as_regime_barrier_1000_runs.csv", index=False)
    df_scmm.to_csv(results_dir / "scmm_v4_regime_barrier_1000_runs.csv", index=False)

    summary_df = pd.DataFrame([
        summarize(df_as, "AS_regime_barrier"),
        summarize(df_scmm, "SCMM_v4_gated_regime_barrier"),
    ])
    summary_df.to_csv(results_dir / "regime_barrier_1000_summary.csv", index=False)

    pvals = {
        "metric": [
            "final_wealth",
            "sharpe_like",
            "mean_abs_inventory",
            "max_drawdown",
            "steps_survived",
            "barrier_hit_rate",
        ],
        "welch_ttest_pvalue": [
            welch_pvalue(df_as["final_wealth"], df_scmm["final_wealth"]),
            welch_pvalue(df_as["sharpe_like"], df_scmm["sharpe_like"]),
            welch_pvalue(df_as["mean_abs_inventory"], df_scmm["mean_abs_inventory"]),
            welch_pvalue(df_as["max_drawdown"], df_scmm["max_drawdown"]),
            welch_pvalue(df_as["steps_survived"], df_scmm["steps_survived"]),
            np.nan,
        ],
        "mannwhitney_pvalue": [
            mannwhitney_pvalue(df_as["final_wealth"], df_scmm["final_wealth"]),
            mannwhitney_pvalue(df_as["sharpe_like"], df_scmm["sharpe_like"]),
            mannwhitney_pvalue(df_as["mean_abs_inventory"], df_scmm["mean_abs_inventory"]),
            mannwhitney_pvalue(df_as["max_drawdown"], df_scmm["max_drawdown"]),
            mannwhitney_pvalue(df_as["steps_survived"], df_scmm["steps_survived"]),
            np.nan,
        ],
        "binary_test_pvalue": [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            barrier_fisher_pvalue(df_as["barrier_hit"], df_scmm["barrier_hit"]),
        ],
    }

    pvals_df = pd.DataFrame(pvals)
    pvals_df.to_csv(results_dir / "regime_barrier_1000_pvalues.csv", index=False)

    print("\nLarge-sample summary:")
    print(summary_df.to_string(index=False))
    print("\nP-values:")
    print(pvals_df.to_string(index=False))


if __name__ == "__main__":
    main()