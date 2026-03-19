from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config.default_params import Params
from src.metrics.performance import compute_all_metrics
from src.simulator.simulator import AvellanedaStoikovSimulator
from src.utils.helpers import ensure_dir


def save_basic_plots(df: pd.DataFrame, plot_dir: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(df["time"], df["mid_price"], label="Mid Price")
    plt.plot(df["time"], df["reservation_price"], label="Reservation Price", linestyle="--")
    plt.plot(df["time"], df["bid"], label="Bid", alpha=0.8)
    plt.plot(df["time"], df["ask"], label="Ask", alpha=0.8)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title("Mid Price, Reservation Price, and Quotes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "price_quotes.png")
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(df["time"], df["inventory"])
    plt.xlabel("Time")
    plt.ylabel("Inventory")
    plt.title("Inventory Trajectory")
    plt.tight_layout()
    plt.savefig(plot_dir / "inventory.png")
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(df["time"], df["wealth"])
    plt.xlabel("Time")
    plt.ylabel("Wealth")
    plt.title("Marked-to-Market Wealth")
    plt.tight_layout()
    plt.savefig(plot_dir / "wealth.png")
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(df["time"], df["half_spread"])
    plt.xlabel("Time")
    plt.ylabel("Half-Spread")
    plt.title("Half-Spread Over Time")
    plt.tight_layout()
    plt.savefig(plot_dir / "half_spread.png")
    plt.close()


def main() -> None:
    params = Params()

    output_dir = ensure_dir("outputs")
    logs_dir = ensure_dir(output_dir / "logs")
    plots_dir = ensure_dir(output_dir / "plots")
    results_dir = ensure_dir(output_dir / "results")

    sim = AvellanedaStoikovSimulator(params)
    df = sim.run()

    df.to_csv(logs_dir / "baseline_run.csv", index=False)
    save_basic_plots(df, plots_dir)

    summary = compute_all_metrics(df)
    pd.DataFrame([summary]).to_csv(results_dir / "baseline_summary.csv", index=False)

    print("Baseline run complete.")
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()