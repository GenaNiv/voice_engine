import argparse
import logging
import signal
import sys
import time
from pathlib import Path

from .config import load_config
from .runtime import VoiceEngineRuntime


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Voice engine service")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/voice_engine.example.yaml"),
        help="Path to YAML config file",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the voice engine service."""
    args = parse_args(argv)
    config = load_config(args.config)

    level_name = config.logging.level.upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(threadName)s] %(message)s",
        force=True,
    )
    logging.info("Loaded config from %s", args.config)

    runtime = VoiceEngineRuntime(config)
    runtime.start()
    logging.info("Voice engine runtime starting.")

    stopping = False

    def _handle_signal(signum, frame):
        nonlocal stopping
        if stopping:
            return
        stopping = True
        logging.info("Received signal %s; stopping runtime", signum)
        runtime.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received; shutting down runtime")
        _handle_signal("keyboard", None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
