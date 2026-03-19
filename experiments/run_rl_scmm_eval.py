from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO

from config.default_params import Params
from src.metrics.performance import compute_all_metrics
from src.rl.env_scmm import SCMMEnv
from src.utils.helpers import ensure_dir


def rollout_one_episode(model: PPO, seed: int) -> tuple[pd.DataFrame, dict]:
    params = Params(seed=seed)
    env = SCMMEnv(
        params=params,
        lambda_q=0.05,
        lambda_d=0.02,
        lambda_a=0.01,
        max_u_skew=0.5,
        max_u_spr=0.5,
        fill_imbalance_window=20,
    )

    obs, info = env.reset(seed=seed)

    rows = []
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        rows.append(
            {
                "time": info["time"],
                "mid_price": info["mid_price"],
                "inventory": info["inventory"],
                "cash": info["cash"],
                "wealth": info["wealth_after"],
                "drawdown": info["drawdown_after"],
                "reservation_price_as": info["reservation_price_as"],
                "half_spread_as": info["half_spread_as"],
                "reservation_price_sc": info["reservation_price_sc"],
                "half_spread_sc": info["half_spread_sc"],
                "u_skew": info["u_skew"],
                "u_spr": info["u_spr"],
                "bid": info["bid"],
                "ask": info["ask"],
                "bid_fill": info["bid_fill"],
                "ask_fill": info["ask_fill"],
                "reward": info["reward"],
            }
        )

    df = pd.DataFrame(rows)
    metrics = compute_all_metrics(df)
    metrics["seed"] = seed

    env.close()
    return df, metrics


def save_basic_plots(df: pd.DataFrame, plot_dir: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(df["time"], df["mid_price"], label="Mid Price")
    plt.plot(df["time"], df["reservation_price_as"], label="AS Reservation", linestyle="--")
    plt.plot(df["time"], df["reservation_price_sc"], label="RL Reservation", linestyle=":")
    plt.plot(df["time"], df["bid"], label="Bid", alpha=0.8)
    plt.plot(df["time"], df["ask"], label="Ask", alpha=0.8)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title("RL-SCMM Quotes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "rl_scmm_price_quotes.png")
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(df["time"], df["inventory"])
    plt.xlabel("Time")
    plt.ylabel("Inventory")
    plt.title("RL-SCMM Inventory")
    plt.tight_layout()
    plt.savefig(plot_dir / "rl_scmm_inventory.png")
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(df["time"], df["wealth"])
    plt.xlabel("Time")
    plt.ylabel("Wealth")
    plt.title("RL-SCMM Wealth")
    plt.tight_layout()
    plt.savefig(plot_dir / "rl_scmm_wealth.png")
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(df["time"], df["u_skew"], label="u_skew")
    plt.plot(df["time"], df["u_spr"], label="u_spr")
    plt.xlabel("Time")
    plt.ylabel("Correction")
    plt.title("RL-SCMM Corrections")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "rl_scmm_corrections.png")
    plt.close()


def main() -> None:
    model_path = "outputs/results/ppo_scmm_v2"
    n_runs = 50
    base_seed = 42

    output_dir = ensure_dir("outputs")
    logs_dir = ensure_dir(output_dir / "logs")
    plots_dir = ensure_dir(output_dir / "plots")
    results_dir = ensure_dir(output_dir / "results")

    model = PPO.load(model_path)

    rows = []
    first_df = None

    for i in range(n_runs):
        seed = base_seed + i
        df, metrics = rollout_one_episode(model, seed)
        rows.append(metrics)

        if i == 0:
            first_df = df
            df.to_csv(logs_dir / "rl_scmm_run.csv", index=False)

    df_runs = pd.DataFrame(rows)
    df_runs.to_csv(results_dir / "multi_run_rl_scmm.csv", index=False)

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

    pd.DataFrame([aggregate]).to_csv(results_dir / "multi_run_rl_scmm_summary.csv", index=False)

    if first_df is not None:
        save_basic_plots(first_df, plots_dir)

    print("\nRL-SCMM aggregate summary:")
    print(pd.DataFrame([aggregate]).to_string(index=False))


if __name__ == "__main__":
    main()