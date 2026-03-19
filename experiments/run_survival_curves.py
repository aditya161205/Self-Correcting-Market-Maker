from __future__ import annotations

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.default_params import Params
from src.simulator.as_regime_barrier_simulator import ASRegimeBarrierSimulator
from src.simulator.scmm_v4_gated_regime_barrier_simulator import SCMMV4GatedRegimeBarrierSimulator
from src.utils.helpers import ensure_dir


def collect_event_times(simulator_cls, params: Params, n_runs: int = 1000, base_seed: int = 42) -> pd.DataFrame:
    rows = []

    for i in range(n_runs):
        seed = base_seed + i
        run_params = replace(params, seed=seed)

        sim = simulator_cls(run_params)
        df = sim.run()

        barrier_hit = int(bool(df["barrier_hit"].iloc[-1]))
        steps_survived = int(len(df) - 1)

        rows.append(
            {
                "seed": seed,
                "event": barrier_hit,          # 1 if barrier hit, 0 if censored/survived
                "time": steps_survived,        # time to event or censoring
            }
        )

    return pd.DataFrame(rows)


def kaplan_meier_curve(events_df: pd.DataFrame) -> pd.DataFrame:
    df = events_df.sort_values("time").reset_index(drop=True)

    unique_times = sorted(df["time"].unique())
    n = len(df)
    at_risk = n
    survival = 1.0

    rows = [{"time": 0, "survival": 1.0}]

    for t in unique_times:
        d_t = int(((df["time"] == t) & (df["event"] == 1)).sum())
        c_t = int(((df["time"] == t) & (df["event"] == 0)).sum())

        if at_risk > 0 and d_t > 0:
            survival *= (1.0 - d_t / at_risk)

        rows.append({"time": int(t), "survival": float(survival)})
        at_risk -= (d_t + c_t)

    return pd.DataFrame(rows)


def main() -> None:
    output_dir = ensure_dir("outputs")
    results_dir = ensure_dir(output_dir / "results")
    plots_dir = ensure_dir(output_dir / "plots")

    params = Params()

    print("Collecting event times for survival curves...")

    df_as_events = collect_event_times(ASRegimeBarrierSimulator, params, n_runs=1000, base_seed=42)
    df_scmm_events = collect_event_times(SCMMV4GatedRegimeBarrierSimulator, params, n_runs=1000, base_seed=42)

    df_as_events.to_csv(results_dir / "as_survival_events.csv", index=False)
    df_scmm_events.to_csv(results_dir / "scmm_survival_events.csv", index=False)

    km_as = kaplan_meier_curve(df_as_events)
    km_scmm = kaplan_meier_curve(df_scmm_events)

    km_as.to_csv(results_dir / "as_survival_curve.csv", index=False)
    km_scmm.to_csv(results_dir / "scmm_survival_curve.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.step(km_as["time"], km_as["survival"], where="post", label="AS")
    plt.step(km_scmm["time"], km_scmm["survival"], where="post", label="SCMM-v4")
    plt.xlabel("Simulation Step")
    plt.ylabel("Survival Probability")
    plt.title("Survival Curves in Markov Regime-Switching Barrier Environment")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "survival_curves_regime_barrier.png")
    plt.close()

    print("\nFinal survival probabilities:")
    print(f"AS final survival probability: {km_as['survival'].iloc[-1]:.4f}")
    print(f"SCMM final survival probability: {km_scmm['survival'].iloc[-1]:.4f}")


if __name__ == "__main__":
    main()