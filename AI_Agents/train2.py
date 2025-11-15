# train2.py
"""
Train a PPO agent for MegaMiner with custom time limit and map selection.
"""

import os
import time
import torch
import argparse
from pathlib import Path
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.vec_env import VecMonitor
from pettingzoo.utils.conversions import aec_to_parallel

# Import MegaMinerEnv from AI_Agents
try:
    import MegaMinerEnv
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent / 'AI_Agents'))
    import MegaMinerEnv


class TimeLimitCallback(BaseCallback):
    """Stops training after a fixed amount of seconds."""
    def __init__(self, max_time: int, verbose: int = 0):
        super().__init__(verbose)
        self.start_time = time.time()
        self.max_time = max_time

    def _on_step(self) -> bool:
        if time.time() - self.start_time > self.max_time:
            if self.verbose:
                print(f"[TimeLimitCallback] Reached max training time ({self.max_time} s). Stopping.")
            return False
        return True


def main(args):
    # --- Device selection ---
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"[INFO] Using device: {device}")

    # --- Environment setup ---
    map_file = Path(__file__).resolve().parent.parent / "maps" / args.map_path
    env = MegaMinerEnv.env(map_path=str(map_file))
    env = aec_to_parallel(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")

    # --- Logging & model directories ---
    log_dir = Path("training/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path("training/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = model_dir / "best_model/best_model.zip"

    # --- Load or create PPO model ---
    if best_model_path.exists():
        print("[INFO] Loading existing model...")
        model = PPO.load(best_model_path, env=env, tensorboard_log=str(log_dir), device=device)
        from stable_baselines3.common import utils
        model.set_logger(utils.configure_logger(verbose=1, tensorboard_log=str(log_dir), reset_num_timesteps=False))
    else:
        print("[INFO] Creating new PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=str(log_dir),
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            device=device,
        )

    # --- Callbacks ---
    time_callback = TimeLimitCallback(max_time=args.train_minutes * 60, verbose=1)

    eval_env = MegaMinerEnv.env(map_path=str(map_file))
    eval_env = aec_to_parallel(eval_env)
    eval_env = ss.pettingzoo_env_to_vec_env_v1(eval_env)
    eval_env = ss.concat_vec_envs_v1(eval_env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / "best_model"),
        log_path=str(log_dir / "eval"),
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    callbacks = CallbackList([time_callback, eval_callback])

    # --- Start training ---
    print(f"[INFO] Training PPO agent for {args.train_minutes} minutes on map {args.map_path}...")
    model.learn(total_timesteps=10_000_000, callback=callbacks)

    # --- Save final model ---
    final_model_path = model_dir / "ppo_megaminer_final"
    model.save(str(final_model_path))
    print(f"[INFO] Final model saved at {final_model_path}")
    print(f"[INFO] Best model saved in {model_dir}/best_model/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent for MegaMiner")
    parser.add_argument("--train-minutes", type=int, default=20, help="Training duration in minutes")
    parser.add_argument("--map-path", type=str, default="map0.json", help="Map file to train on")
    args = parser.parse_args()

    main(args)