import json
import logging
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "agents.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("research_system")


def log_agent_call(agent_name: str) -> Callable:
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(state: dict, *args, **kwargs) -> Any:
            start = time.perf_counter()
            entry = {
                "agent": agent_name,
                "timestamp": datetime.utcnow().isoformat(),
                "input_query": state.get("query"),
                "iteration": state.get("iterations", 0),
            }
            logger.info(f"[{agent_name}] started | iteration={entry['iteration']}")

            try:
                result = fn(state, *args, **kwargs)
                elapsed = round(time.perf_counter() - start, 3)
                entry["elapsed_seconds"] = elapsed
                entry["status"] = "success"
                logger.info(f"[{agent_name}] completed in {elapsed}s")
                _append_log(entry)
                return result
            except Exception as exc:
                elapsed = round(time.perf_counter() - start, 3)
                entry["elapsed_seconds"] = elapsed
                entry["status"] = "error"
                entry["error"] = str(exc)
                logger.error(f"[{agent_name}] failed after {elapsed}s: {exc}")
                _append_log(entry)
                raise

        return wrapper
    return decorator


def _append_log(entry: dict) -> None:
    log_file = LOG_DIR / "agent_runs.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")
