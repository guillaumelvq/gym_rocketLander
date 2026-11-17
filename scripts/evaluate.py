"""Evaluate a trained agent and record a video.

    uv run python scripts/evaluate.py runs/<run>/best_model.zip --level 0 --video landing.mp4
"""

import argparse

from rocketlander.training import ALGOS, evaluate, record_episode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--algo", default="PPO", choices=list(ALGOS))
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--discrete", action="store_true")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--video", default=None, help="output .mp4 or .gif")
    args = p.parse_args()

    model = ALGOS[args.algo].load(args.model, device="cpu")
    print(evaluate(model, args.level, not args.discrete, n_episodes=args.episodes))
    if args.video:
        path, _ = record_episode(model, args.video, args.level, not args.discrete)
        print("video:", path)


if __name__ == "__main__":
    main()
