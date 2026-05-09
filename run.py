#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))

from graph.research_graph import research_graph
from observability.logger import logger


def run(query: str, verbose: bool = False) -> None:
    print(f"\n🔍 Query: {query}\n{'─' * 60}")

    initial_state = {
        "query": query,
        "search_results": [],
        "summary": None,
        "critique": None,
        "iterations": 0,
        "final_answer": "",
        "agent_logs": [],
        "error": None,
    }

    start = time.perf_counter()

    for event in research_graph.stream(initial_state):
        for node_name, output in event.items():
            print(f"  ▶ [{node_name}]", end=" ")
            logs = output.get("agent_logs", [])
            if logs:
                last = logs[-1]
                details = {k: v for k, v in last.items() if k != "agent"}
                print(json.dumps(details, indent=None))
            else:
                print()

    elapsed = round(time.perf_counter() - start, 2)
    final_state = research_graph.invoke(initial_state) if False else None

    print(f"\n{'─' * 60}")
    print(f"✅ Completed in {elapsed}s\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Agent Research Assistant CLI")
    parser.add_argument("query", nargs="?", help="Research query")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if not args.query:
        print("Usage: python run.py \"your research query here\"")
        sys.exit(1)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)

    initial_state = {
        "query": args.query,
        "search_results": [],
        "summary": None,
        "critique": None,
        "iterations": 0,
        "final_answer": "",
        "agent_logs": [],
        "error": None,
    }

    start = time.perf_counter()
    result = research_graph.invoke(initial_state)
    elapsed = round(time.perf_counter() - start, 2)

    print(f"\n{'═' * 60}")
    print(result.get("final_answer", "No answer generated."))
    print(f"\n{'─' * 60}")
    print(f"Iterations: {result.get('iterations', 0)} | "
          f"Claims: {len(result['summary'].claims) if result.get('summary') else 0} | "
          f"Time: {elapsed}s")


if __name__ == "__main__":
    main()
