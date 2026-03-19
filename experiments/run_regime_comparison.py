import pandas as pd

from config.default_params import Params
from src.metrics.performance import compute_all_metrics
from src.simulator.as_regime_simulator import ASRegimeSwitchingSimulator
from src.simulator.scmm_v4_gated_regime_simulator import SCMMV4GatedRegimeSimulator
from src.utils.helpers import ensure_dir


def run_model(simulator_cls, params: Params, n_runs: int = 50, base_seed: int = 42) -> dict:
    rows = []

    for i in range(n_runs):
        seed = base_seed + i
        run_params = Params(**params.__dict__)
        run_params.seed = seed

        sim = simulator_cls(run_params)
        df = sim.run()
        metrics = compute_all_metrics(df)
        rows.append(metrics)

    df_runs = pd.DataFrame(rows)

    return {
        "n_runs": n_runs,
        "mean_final_wealth": float(df_runs["final_wealth"].mean()),
        "std_final_wealth": float(df_runs["final_wealth"].std()),
        "mean_sharpe_like": float(df_runs["sharpe_like"].mean()),
        "mean_abs_inventory": float(df_runs["mean_abs_inventory"].mean()),
        "mean_inventory_std": float(df_runs["inventory_std"].mean()),
        "mean_max_drawdown": float(df_runs["max_drawdown"].mean()),
        "mean_mean_drawdown": float(df_runs["mean_drawdown"].mean()),
        "mean_fill_imbalance": float(df_runs["fill_imbalance"].mean()),
        "mean_total_fills": float(df_runs["total_fills"].mean()),
        "final_wealth_p05": float(df_runs["final_wealth"].quantile(0.05)),
    }


def main() -> None:
    output_dir = ensure_dir("outputs")
    results_dir = ensure_dir(output_dir / "results")

    params = Params()

    as_summary = run_model(ASRegimeSwitchingSimulator, params, n_runs=50, base_seed=42)
    as_summary["model"] = "AS_regime"

    scmm_summary = run_model(SCMMV4GatedRegimeSimulator, params, n_runs=50, base_seed=42)
    scmm_summary["model"] = "SCMM_v4_gated_regime"

    df = pd.DataFrame([as_summary, scmm_summary])

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
    ]
    df = df[cols]

    df.to_csv(results_dir / "regime_comparison_summary.csv", index=False)

    print("\nRegime comparison summary:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()