"""Train an agent on RocketLander.

    uv run python scripts/train.py --level 0 --timesteps 500000
    uv run tensorboard --logdir runs
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.vec_env import VecNormalize

from rocketlander.challenges import CodeReward
from rocketlander.env import make_env
from rocketlander.training import ALGOS, LandingStatsCallback, evaluate, make_vec


def _probe_info() -> dict:
    env = make_env(level=0)
    _, info = env.reset(seed=0)
    env.close()
    return dict(info, memory={}, base_reward=0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="PPO", choices=list(ALGOS))
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--discrete", action="store_true")
    p.add_argument("--timesteps", type=float, default=5e5)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--no-reward-norm", action="store_true", help="disable VecNormalize on rewards")
    p.add_argument("--progress-bar", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--config", default="{}", help='JSON scenario overrides, e.g. {"start_height": 0.5}')
    p.add_argument("--reward-code", default=None,
                   help="Python file defining reward(s, base_reward) (see rocketlander/challenges.py)")
    p.add_argument("--reward", default="{}", help='JSON overrides of the reward weights, e.g. {"shaping": 3}')
    args = p.parse_args()

    continuous = not args.discrete
    name = args.name or f"{args.algo}_level{args.level}_{'cont' if continuous else 'disc'}_{datetime.now():%Y%m%d-%H%M%S}"
    out = Path("runs") / name
    rp = json.loads(args.reward)
    cfg = json.loads(args.config)
    out.mkdir(parents=True, exist_ok=True)
    reward_fn = None
    if args.reward_code:
        code = Path(args.reward_code).read_text(encoding="utf8")
        if Path(args.reward_code).resolve() != (out / "reward.py").resolve():
            (out / "reward.py").write_text(code, encoding="utf8")
        reward_fn = CodeReward(code)
        reward_fn(_probe_info(), 0.0)  # fail fast on syntax / key errors
    (out / "meta.json").write_text(json.dumps({
        "name": name, "algo": args.algo, "level": args.level, "continuous": continuous,
        "timesteps": int(args.timesteps), "reward": rp, "config": cfg, "seed": args.seed,
        "custom_reward": bool(args.reward_code),
        "gamma": args.gamma, "frame_skip": args.frame_skip, "reward_norm": not args.no_reward_norm,
        "created": datetime.now().isoformat(timespec="seconds"),
    }, indent=1), encoding="utf8")
    env = make_vec(args.level, continuous, n_envs=args.n_envs, seed=args.seed, reward_params=rp,
                   frame_skip=args.frame_skip, config=cfg, reward_fn=reward_fn)
    eval_env = make_vec(args.level, continuous, n_envs=1, seed=args.seed + 1000, reward_params=rp,
                        frame_skip=args.frame_skip, config=cfg, reward_fn=reward_fn)

    if not args.no_reward_norm:
        # Rewards span -5..+15 per episode end and ~0.01 per step: scaling them keeps the
        # value function well conditioned (as in RL Zoo for LunarLanderContinuous).
        # Only training rewards are scaled; the saved policy needs no statistics.
        env = VecNormalize(env, norm_obs=False, norm_reward=True, gamma=args.gamma)
        # EvalCallback requires the same wrapper on the eval env (here a no-op)
        eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False, training=False)
    # PPO hyperparameters from RL Zoo (LunarLanderContinuous), a very similar task
    kwargs = dict(n_steps=1024, batch_size=64, n_epochs=4, gamma=args.gamma, gae_lambda=0.98,
                  ent_coef=0.01, learning_rate=3e-4) if args.algo == "PPO" else dict(gamma=args.gamma)
    model = ALGOS[args.algo]("MlpPolicy", env, seed=args.seed, verbose=0,
                             tensorboard_log=str(Path("runs")), **kwargs)
    model.learn(
        total_timesteps=int(args.timesteps),
        tb_log_name=name,
        progress_bar=args.progress_bar,
        callback=CallbackList([
            LandingStatsCallback(progress_path=out / "progress.json", total_timesteps=int(args.timesteps)),
            EvalCallback(eval_env, best_model_save_path=str(out), eval_freq=max(25_000 // args.n_envs, 1),
                         n_eval_episodes=20, deterministic=True, verbose=0),
        ]),
    )
    model.save(out / "last_model")
    print("saved to", out)
    scores = evaluate(model, args.level, continuous, reward_params=rp, frame_skip=args.frame_skip, config=cfg,
                      reward_fn=reward_fn)
    (out / "eval.json").write_text(json.dumps(scores, indent=1), encoding="utf8")
    print("eval (last model, 100 episodes):", scores)


if __name__ == "__main__":
    main()
