from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .models import TaskStatus
from .orchestrator import Config, Orchestrator
from .workspace import Workspace


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="nanoteam",
        description="Multi-agent orchestration for complex coding tasks",
    )
    parser.add_argument("goal", nargs="?", help="High-level objective")
    parser.add_argument("--resume", action="store_true", help="Resume from .nanoteam/ state")
    parser.add_argument("--status", action="store_true", help="Print task graph status")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root directory")
    parser.add_argument("--lead-model", default="opus", help="Model for Lead agent")
    parser.add_argument("--worker-model", default="sonnet", help="Default model for workers")
    parser.add_argument("--max-budget", type=float, default=10.0, help="Total budget in USD")
    parser.add_argument(
        "--checkpoint", default="plan,finish",
        help="Checkpoint timing: comma-separated list of plan,phase,finish,none (default: plan,finish)",
    )

    args = parser.parse_args()
    ws = Workspace(args.root)

    if args.status:
        _print_status(ws)
        return

    if args.resume:
        if not (ws.base / "task_graph.json").exists():
            print("No .nanoteam/ state found. Start with a goal first.", file=sys.stderr)
            sys.exit(1)
        graph = ws.load_task_graph()
        # Reset any IN_PROGRESS tasks back to READY (interrupted run)
        for task in graph.tasks.values():
            if task.status == TaskStatus.IN_PROGRESS:
                task.status = TaskStatus.READY
        ws.save_task_graph(graph)
    else:
        if not args.goal:
            parser.error("Provide a goal or use --resume")
        ws.init(args.goal)

    checkpoints = set()
    if args.checkpoint != "none":
        checkpoints = {c.strip() for c in args.checkpoint.split(",") if c.strip()}

    config = Config(
        lead_model=args.lead_model,
        worker_model=args.worker_model,
        max_budget=args.max_budget,
        checkpoints=checkpoints,
    )

    orch = Orchestrator(ws, config)
    try:
        orch.run()
    except RuntimeError as e:
        print(f"[nanoteam] Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[nanoteam] Interrupted. Use --resume to continue.", file=sys.stderr)
        sys.exit(130)


def _print_status(ws: Workspace) -> None:
    if not (ws.base / "task_graph.json").exists():
        print("No .nanoteam/ state found.")
        return

    graph = ws.load_task_graph()
    print(f"Goal: {graph.goal}")
    print(f"Tasks: {len(graph.tasks)}")
    print(f"Roles: {', '.join(graph.roles.keys()) or 'none'}")
    print()

    status_counts: dict[str, int] = {}
    for task in graph.tasks.values():
        status_counts[task.status.value] = status_counts.get(task.status.value, 0) + 1
        marker = {"done": "+", "failed": "x", "in_progress": ">", "ready": "~", "review": "?"}
        icon = marker.get(task.status.value, " ")
        print(f"  [{icon}] {task.id}: {task.title} ({task.status.value})")

    print()
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")


if __name__ == "__main__":
    main()
