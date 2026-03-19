from __future__ import annotations

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from config.default_params import Params
from src.rl.env_scmm_v5 import SCMMV5Env
from src.utils.helpers import ensure_dir


def main() -> None:
    output_dir = ensure_dir("outputs")
    results_dir = ensure_dir(output_dir / "results")

    env = SCMMV5Env(
        params=Params(),
        lambda_q=0.05,
        lambda_d=0.02,
        lambda_res=0.02,
        residual_skew_max=0.10,
        residual_spr_max=0.05,
    )

    check_env(env, warn=True, skip_render_check=True)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        device="auto",
    )

    model.learn(total_timesteps=250_000)
    model.save(results_dir / "ppo_scmm_v5")

    print("SCMM-v5 PPO training complete. Model saved to outputs/results/ppo_scmm_v5.zip")


if __name__ == "__main__":
    main()