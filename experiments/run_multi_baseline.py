from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config.default_params import Params
from src.metrics.performance import compute_all_metrics
from src.simulator.simulator import AvellanedaStoikovSimulator
from src.utils.helpers import ensure_dir


def run_one(seed: int) -> dict:
    params = Params(seed=seed)
    sim = AvellanedaStoikovSimulator(params)
    df = sim.run()

    metrics = compute_all_metrics(df)
    metrics["seed"] = seed
    return metrics


def save_histogram(df_runs: pd.DataFrame, plot_dir: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.hist(df_runs["final_wealth"], bins=15)
    plt.xlabel("Final Wealth")
    plt.ylabel("Frequency")
    plt.title("Distribution of Final Wealth Across Runs")
    plt.tight_layout()
    plt.savefig(plot_dir / "multi_run_final_wealth_hist.png")
    plt.close()


def save_drawdown_histogram(df_runs: pd.DataFrame, plot_dir: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.hist(df_runs["max_drawdown"], bins=15)
    plt.xlabel("Max Drawdown")
    plt.ylabel("Frequency")
    plt.title("Distribution of Max Drawdown Across Runs")
    plt.tight_layout()
    plt.savefig(plot_dir / "multi_run_max_drawdown_hist.png")
    plt.close()


def main() -> None:
    n_runs = 50
    base_seed = 42

    output_dir = ensure_dir("outputs")
    results_dir = ensure_dir(output_dir / "results")
    plots_dir = ensure_dir(output_dir / "plots")

    rows = []
    for i in range(n_runs):
        seed = base_seed + i
        rows.append(run_one(seed))

    df_runs = pd.DataFrame(rows)
    df_runs.to_csv(results_dir / "multi_run_baseline.csv", index=False)

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

    df_agg = pd.DataFrame([aggregate])
    df_agg.to_csv(results_dir / "multi_run_baseline_summary.csv", index=False)

    save_histogram(df_runs, plots_dir)
    save_drawdown_histogram(df_runs, plots_dir)

    print("\nPer-run results saved to:")
    print(results_dir / "multi_run_baseline.csv")

    print("\nAggregate summary saved to:")
    print(results_dir / "multi_run_baseline_summary.csv")

    print("\nAggregate summary:")
    print(df_agg.to_string(index=False))


if __name__ == "__main__":
    main()