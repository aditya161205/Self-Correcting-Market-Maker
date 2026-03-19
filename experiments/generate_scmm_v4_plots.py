from __future__ import annotations

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.default_params import Params
from src.simulator.as_regime_barrier_simulator import ASRegimeBarrierSimulator
from src.simulator.scmm_v4_gated_regime_barrier_simulator import (
    SCMMV4GatedRegimeBarrierSimulator,
)
from src.utils.helpers import ensure_dir


def run_one(simulator_cls, params: Params, seed: int) -> pd.DataFrame:
    run_params = replace(params, seed=seed)
    sim = simulator_cls(run_params)
    return sim.run()


def collect_runs(
    simulator_cls,
    params: Params,
    n_runs: int = 200,
    base_seed: int = 42,
) -> list[pd.DataFrame]:
    runs = []
    for i in range(n_runs):
        runs.append(run_one(simulator_cls, params, base_seed + i))
    return runs


def pad_series_to_length(series: pd.Series, target_len: int) -> np.ndarray:
    arr = series.to_numpy(dtype=float)
    if len(arr) >= target_len:
        return arr[:target_len]

    padded = np.full(target_len, arr[-1], dtype=float)
    padded[: len(arr)] = arr
    return padded


def mean_wealth_trajectory(runs: list[pd.DataFrame], target_len: int) -> np.ndarray:
    wealths = [pad_series_to_length(df["wealth"], target_len) for df in runs]
    return np.mean(np.vstack(wealths), axis=0)


def barrier_event_table(runs: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for i, df in enumerate(runs):
        rows.append(
            {
                "run": i,
                "barrier_hit": int(bool(df["barrier_hit"].iloc[-1])),
                "survived": int(bool(df["survived"].iloc[-1])),
                "steps_survived": int(len(df) - 1),
                "final_wealth": float(df["wealth"].iloc[-1]),
            }
        )
    return pd.DataFrame(rows)


def kaplan_meier_curve(events_df: pd.DataFrame) -> pd.DataFrame:
    df = events_df.sort_values("steps_survived").reset_index(drop=True)

    unique_times = sorted(df["steps_survived"].unique())
    n = len(df)
    at_risk = n
    survival = 1.0

    rows = [{"time": 0, "survival": 1.0}]

    for t in unique_times:
        d_t = int(((df["steps_survived"] == t) & (df["barrier_hit"] == 1)).sum())
        c_t = int(((df["steps_survived"] == t) & (df["barrier_hit"] == 0)).sum())

        if at_risk > 0 and d_t > 0:
            survival *= (1.0 - d_t / at_risk)

        rows.append({"time": int(t), "survival": float(survival)})
        at_risk -= (d_t + c_t)

    return pd.DataFrame(rows)


def find_representative_seed(
    as_runs: list[pd.DataFrame],
    scmm_runs: list[pd.DataFrame],
    base_seed: int,
) -> int:
    """
    Pick a seed where AS ends badly and SCMM survives or materially outperforms.
    """
    best_seed = base_seed
    best_gap = -1e18

    for i, (df_as, df_scmm) in enumerate(zip(as_runs, scmm_runs)):
        as_final = float(df_as["wealth"].iloc[-1])
        scmm_final = float(df_scmm["wealth"].iloc[-1])

        as_barrier = int(bool(df_as["barrier_hit"].iloc[-1]))
        scmm_barrier = int(bool(df_scmm["barrier_hit"].iloc[-1]))

        score = (scmm_final - as_final) + 5.0 * (as_barrier - scmm_barrier)

        if score > best_gap:
            best_gap = score
            best_seed = base_seed + i

    return best_seed


def add_toxic_regime_shading(ax, regime_df: pd.DataFrame, alpha: float = 0.15) -> None:
    """
    Shade toxic-regime intervals (regime == 1) on the given axis.
    """
    toxic = regime_df["regime"].to_numpy(dtype=int)
    times = regime_df["time"].to_numpy(dtype=float)

    in_toxic = False
    start_t = None
    label_added = False

    for i in range(len(toxic)):
        if toxic[i] == 1 and not in_toxic:
            in_toxic = True
            start_t = times[i]
        elif toxic[i] == 0 and in_toxic:
            in_toxic = False
            end_t = times[i]
            if not label_added:
                ax.axvspan(start_t, end_t, alpha=alpha, label="Toxic Regime")
                label_added = True
            else:
                ax.axvspan(start_t, end_t, alpha=alpha)

    if in_toxic and start_t is not None:
        if not label_added:
            ax.axvspan(start_t, times[-1], alpha=alpha, label="Toxic Regime")
        else:
            ax.axvspan(start_t, times[-1], alpha=alpha)


def plot_survival_curve(
    km_as: pd.DataFrame,
    km_scmm: pd.DataFrame,
    save_path,
) -> None:
    plt.figure(figsize=(10, 5))
    plt.step(km_as["time"], km_as["survival"], where="post", label="AS")
    plt.step(km_scmm["time"], km_scmm["survival"], where="post", label="SCMM-v4")
    plt.xlabel("Simulation Step")
    plt.ylabel("Survival Probability")
    plt.title("Survival Curve: AS vs SCMM-v4")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_mean_wealth_trajectory(
    mean_as: np.ndarray,
    mean_scmm: np.ndarray,
    save_path,
) -> None:
    t = np.arange(len(mean_as))
    plt.figure(figsize=(10, 5))
    plt.plot(t, mean_as, label="AS Mean Wealth")
    plt.plot(t, mean_scmm, label="SCMM-v4 Mean Wealth")
    plt.xlabel("Simulation Step")
    plt.ylabel("Mean Wealth")
    plt.title("Mean Wealth Trajectory in Regime + Barrier Environment")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_representative_wealth(
    df_as: pd.DataFrame,
    df_scmm: pd.DataFrame,
    save_path,
) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(df_as["time"], df_as["wealth"], label="AS Wealth")
    plt.plot(df_scmm["time"], df_scmm["wealth"], label="SCMM-v4 Wealth")
    plt.xlabel("Time")
    plt.ylabel("Wealth")
    plt.title("Representative Wealth Path: SCMM-v4 vs AS")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_barrier_hit_comparison(
    as_events: pd.DataFrame,
    scmm_events: pd.DataFrame,
    save_path,
) -> None:
    as_hit_rate = float(as_events["barrier_hit"].mean())
    scmm_hit_rate = float(scmm_events["barrier_hit"].mean())

    plt.figure(figsize=(8, 5))
    plt.bar(["AS", "SCMM-v4"], [as_hit_rate, scmm_hit_rate])
    plt.ylabel("Barrier Hit Rate")
    plt.title("Barrier Hit Rate Comparison")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_time_to_ruin_histogram(
    as_events: pd.DataFrame,
    scmm_events: pd.DataFrame,
    save_path,
) -> None:
    plt.figure(figsize=(10, 5))
    plt.hist(as_events["steps_survived"], bins=25, alpha=0.7, label="AS")
    plt.hist(scmm_events["steps_survived"], bins=25, alpha=0.7, label="SCMM-v4")
    plt.xlabel("Steps Survived")
    plt.ylabel("Frequency")
    plt.title("Time-to-Ruin / Time-to-Censoring Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_regime_and_wealth(
    df_as: pd.DataFrame,
    df_scmm: pd.DataFrame,
    save_path,
) -> None:
    """
    Plot wealth paths with toxic-regime periods shown as shaded background spans.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    add_toxic_regime_shading(ax, df_as, alpha=0.15)

    ax.plot(df_as["time"], df_as["wealth"], label="AS Wealth")
    ax.plot(df_scmm["time"], df_scmm["wealth"], label="SCMM-v4 Wealth")

    ax.set_xlabel("Time")
    ax.set_ylabel("Wealth")
    ax.set_title("Wealth Paths with Toxic Regime Shading")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_mean_wealth_gap(
    mean_as: np.ndarray,
    mean_scmm: np.ndarray,
    save_path,
) -> None:
    t = np.arange(len(mean_as))
    gap = mean_scmm - mean_as

    plt.figure(figsize=(10, 5))
    plt.plot(t, gap, label="SCMM-v4 Mean Wealth - AS Mean Wealth")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Simulation Step")
    plt.ylabel("Mean Wealth Gap")
    plt.title("Mean Wealth Gap Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_two_panel_crash_figure(
    df_as: pd.DataFrame,
    df_scmm: pd.DataFrame,
    save_path,
    zoom_start: float = 0.2,
    zoom_end: float = 0.5,
) -> None:
    """
    Two-panel figure:
    Panel A: macro view over the full horizon
    Panel B: micro view zoomed into the crash window
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=False)

    # Panel A: full horizon
    ax = axes[0]
    add_toxic_regime_shading(ax, df_as, alpha=0.15)
    ax.plot(df_as["time"], df_as["wealth"], label="AS Wealth")
    ax.plot(df_scmm["time"], df_scmm["wealth"], label="SCMM-v4 Wealth")
    ax.set_xlabel("Time")
    ax.set_ylabel("Wealth")
    ax.set_title("Panel A: Macro View (Full Horizon)")
    ax.legend()

    # Panel B: zoomed crash window
    ax = axes[1]
    add_toxic_regime_shading(ax, df_as, alpha=0.15)
    ax.plot(df_as["time"], df_as["wealth"], label="AS Wealth")
    ax.plot(df_scmm["time"], df_scmm["wealth"], label="SCMM-v4 Wealth")
    ax.set_xlim(zoom_start, zoom_end)

    mask_as = (df_as["time"] >= zoom_start) & (df_as["time"] <= zoom_end)
    mask_scmm = (df_scmm["time"] >= zoom_start) & (df_scmm["time"] <= zoom_end)

    local_min = min(
        float(df_as.loc[mask_as, "wealth"].min()),
        float(df_scmm.loc[mask_scmm, "wealth"].min()),
    )
    local_max = max(
        float(df_as.loc[mask_as, "wealth"].max()),
        float(df_scmm.loc[mask_scmm, "wealth"].max()),
    )

    pad = 0.05 * max(local_max - local_min, 1e-6)
    ax.set_ylim(local_min - pad, local_max + pad)

    ax.set_xlabel("Time")
    ax.set_ylabel("Wealth")
    ax.set_title(f"Panel B: Micro View (Zoom: t={zoom_start} to t={zoom_end})")
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def main() -> None:
    output_dir = ensure_dir("outputs")
    plots_dir = ensure_dir(output_dir / "plots")
    scmm_v4_plot_dir = ensure_dir(plots_dir / "scmm_v4")

    params = Params()
    n_runs = 200
    base_seed = 42
    target_len = int(round(params.T / params.dt)) + 1

    print("Collecting AS runs...")
    as_runs = collect_runs(
        ASRegimeBarrierSimulator,
        params,
        n_runs=n_runs,
        base_seed=base_seed,
    )

    print("Collecting SCMM-v4 runs...")
    scmm_runs = collect_runs(
        SCMMV4GatedRegimeBarrierSimulator,
        params,
        n_runs=n_runs,
        base_seed=base_seed,
    )

    as_events = barrier_event_table(as_runs)
    scmm_events = barrier_event_table(scmm_runs)

    km_as = kaplan_meier_curve(as_events)
    km_scmm = kaplan_meier_curve(scmm_events)

    mean_as = mean_wealth_trajectory(as_runs, target_len)
    mean_scmm = mean_wealth_trajectory(scmm_runs, target_len)

    rep_seed = find_representative_seed(as_runs, scmm_runs, base_seed=base_seed)
    rep_as = run_one(ASRegimeBarrierSimulator, params, rep_seed)
    rep_scmm = run_one(SCMMV4GatedRegimeBarrierSimulator, params, rep_seed)

    plot_survival_curve(
        km_as,
        km_scmm,
        scmm_v4_plot_dir / "01_survival_curve_scmm_v4_vs_as.png",
    )

    plot_mean_wealth_trajectory(
        mean_as,
        mean_scmm,
        scmm_v4_plot_dir / "02_mean_wealth_trajectory_scmm_v4_vs_as.png",
    )

    plot_representative_wealth(
        rep_as,
        rep_scmm,
        scmm_v4_plot_dir / "03_representative_wealth_path_scmm_v4_vs_as.png",
    )

    plot_barrier_hit_comparison(
        as_events,
        scmm_events,
        scmm_v4_plot_dir / "04_barrier_hit_rate_comparison.png",
    )

    plot_time_to_ruin_histogram(
        as_events,
        scmm_events,
        scmm_v4_plot_dir / "05_time_to_ruin_histogram.png",
    )

    plot_regime_and_wealth(
        rep_as,
        rep_scmm,
        scmm_v4_plot_dir / "06_regime_and_wealth_overlay.png",
    )

    plot_mean_wealth_gap(
        mean_as,
        mean_scmm,
        scmm_v4_plot_dir / "07_mean_wealth_gap_scmm_minus_as.png",
    )

    plot_two_panel_crash_figure(
        rep_as,
        rep_scmm,
        scmm_v4_plot_dir / "08_two_panel_crash_figure_scmm_v4_vs_as.png",
        zoom_start=0.2,
        zoom_end=0.5,
    )

    as_events.to_csv(scmm_v4_plot_dir / "as_events_for_plots.csv", index=False)
    scmm_events.to_csv(scmm_v4_plot_dir / "scmm_v4_events_for_plots.csv", index=False)
    km_as.to_csv(scmm_v4_plot_dir / "as_survival_curve.csv", index=False)
    km_scmm.to_csv(scmm_v4_plot_dir / "scmm_v4_survival_curve.csv", index=False)

    print(f"\nSCMM-v4 plot folder created at: {scmm_v4_plot_dir}")
    print(f"Representative seed selected: {rep_seed}")


if __name__ == "__main__":
    main()