from __future__ import annotations

import pandas as pd
from stable_baselines3 import PPO

from config.default_params import Params
from src.metrics.performance import compute_all_metrics
from src.rl.env_scmm_v5 import SCMMV5Env
from src.utils.helpers import ensure_dir


def rollout_one_episode(model: PPO, seed: int) -> dict:
    env = SCMMV5Env(
        params=Params(seed=seed),
        lambda_q=0.05,
        lambda_d=0.02,
        lambda_res=0.02,
        residual_skew_max=0.10,
        residual_spr_max=0.05,
    )

    obs, _ = env.reset(seed=seed)
    done = False
    rows = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        rows.append(
            {
                "time": info["time"],
                "mid_price": info["mid_price"],
                "inventory": info["inventory"],
                "wealth": info["wealth_after"],
                "drawdown": info["drawdown_after"],
                "bid_fill": info["bid_fill"],
                "ask_fill": info["ask_fill"],
                "bid": info["bid"],
                "ask": info["ask"],
                "barrier_hit": info["barrier_hit"],
                "survived": info["survived"],
                "regime": info["regime"],
                "u_skew_v4": info["u_skew_v4"],
                "u_spr_v4": info["u_spr_v4"],
                "du_skew_rl": info["du_skew_rl"],
                "du_spr_rl": info["du_spr_rl"],
                "u_skew_total": info["u_skew_total"],
                "u_spr_total": info["u_spr_total"],
                "reward": info["reward"],
            }
        )

    df = pd.DataFrame(rows)
    metrics = compute_all_metrics(df)
    metrics["barrier_hit"] = int(bool(df["barrier_hit"].iloc[-1]))
    metrics["survived"] = int(bool(df["survived"].iloc[-1]))
    metrics["steps_survived"] = int(len(df))
    return metrics


def main() -> None:
    output_dir = ensure_dir("outputs")
    results_dir = ensure_dir(output_dir / "results")

    model = PPO.load("outputs/results/ppo_scmm_v5")

    rows = []
    n_runs = 200
    base_seed = 42

    for i in range(n_runs):
        rows.append(rollout_one_episode(model, base_seed + i))

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "multi_run_scmm_v5_rl.csv", index=False)

    summary = {
        "n_runs": n_runs,
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

    pd.DataFrame([summary]).to_csv(results_dir / "multi_run_scmm_v5_rl_summary.csv", index=False)

    print("\nSCMM-v5 RL summary:")
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()