"""uv run python -m rocketlander.app [--port 8000]"""

import argparse
import webbrowser

import uvicorn


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()
    url = f"http://127.0.0.1:{args.port}"
    print(f"Rocket RL Lab: {url}")
    if not args.no_browser:
        webbrowser.open(url)
    # localhost only: the app runs the reward code typed in the page
    uvicorn.run("rocketlander.app.server:app", host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
