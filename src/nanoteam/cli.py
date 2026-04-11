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
    parser.add_argument("--diagnose", action="store_true", help="AI-assisted diagnosis of current state")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root directory")
    parser.add_argument("--lead-model", default="claude-opus-4-6", help="Model for Lead agent")
    parser.add_argument("--worker-model", default="claude-opus-4-6", help="Default model for workers")
    parser.add_argument("--lead-effort", default="high", choices=["low", "medium", "high", "max"], help="Effort level for Lead")
    parser.add_argument("--worker-effort", default="medium", choices=["low", "medium", "high", "max"], help="Effort level for workers")
    parser.add_argument("--max-budget", type=float, default=10.0, help="Total budget in USD")
    parser.add_argument("--timeout", type=int, default=3600, help="Per-invocation wall-clock timeout in seconds (default: 3600)")
    parser.add_argument("--stall-timeout", type=int, default=300, help="Kill if no output for this many seconds (default: 300)")
    parser.add_argument(
        "--checkpoint", default="plan,finish",
        help="Checkpoint timing: comma-separated list of plan,phase,finish,none (default: plan,finish)",
    )
    parser.add_argument("--skip", nargs="*", metavar="TASK_ID", help="Mark these tasks as done when resuming")
    parser.add_argument("--from", dest="from_task", metavar="TASK_ID", help="Resume from this task (mark all prior as done)")

    args = parser.parse_args()

    # Guard against common mistake: `nanoteam resume` instead of `nanoteam --resume`
    FLAG_LIKE_GOALS = {"resume", "status", "diagnose"}
    if args.goal and args.goal.lower() in FLAG_LIKE_GOALS:
        print(
            f'[nanoteam] Did you mean --{args.goal.lower()}? '
            f'Use "nanoteam --{args.goal.lower()}" instead of "nanoteam {args.goal}".',
            file=sys.stderr,
        )
        sys.exit(1)

    ws = Workspace(args.root)

    if args.status:
        _print_status(ws)
        return

    if args.diagnose:
        _diagnose(ws, args)
        return

    if args.resume:
        if not (ws.base / "task_graph.json").exists():
            print("No .nanoteam/ state found. Start with a goal first.", file=sys.stderr)
            sys.exit(1)
        graph = ws.load_task_graph()
        # IN_PROGRESS tasks with session_ids will be resumed by the orchestrator

        # --skip: force specific tasks to DONE
        if args.skip:
            for tid in args.skip:
                if tid in graph.tasks:
                    graph.tasks[tid].status = TaskStatus.DONE
                    print(f"[nanoteam] Skipped {tid}: {graph.tasks[tid].title}", file=sys.stderr)
                else:
                    print(f"[nanoteam] Warning: task {tid} not found", file=sys.stderr)

        # --from: mark everything before this task as DONE
        if args.from_task:
            if args.from_task not in graph.tasks:
                print(f"[nanoteam] Error: task {args.from_task} not found", file=sys.stderr)
                sys.exit(1)
            target_deps = _collect_all_deps(graph.tasks, args.from_task)
            for tid in target_deps:
                if graph.tasks[tid].status != TaskStatus.DONE:
                    graph.tasks[tid].status = TaskStatus.DONE
                    print(f"[nanoteam] Auto-completed {tid}: {graph.tasks[tid].title}", file=sys.stderr)

        ws.save_task_graph(graph)
    else:
        if not args.goal:
            parser.error("Provide a goal or use --resume")
        ws.init(args.goal)

    # Build config: load saved config on resume, CLI args override
    saved = ws.load_config() if args.resume else None
    checkpoints = set()
    if args.checkpoint != "none":
        checkpoints = {c.strip() for c in args.checkpoint.split(",") if c.strip()}

    if saved:
        config = Config.from_dict(saved)
        # CLI explicit overrides (only if user actually passed them)
        _apply_cli_overrides(config, args, parser, checkpoints)
    else:
        config = Config(
            lead_model=args.lead_model,
            worker_model=args.worker_model,
            lead_effort=args.lead_effort,
            worker_effort=args.worker_effort,
            max_budget=args.max_budget,
            timeout=args.timeout,
            stall_timeout=args.stall_timeout,
            checkpoints=checkpoints,
        )

    ws.save_config(config.to_dict())

    orch = Orchestrator(ws, config)
    try:
        orch.run()
    except RuntimeError as e:
        print(f"[nanoteam] Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[nanoteam] Interrupted. Use --resume to continue.", file=sys.stderr)
        sys.exit(130)


def _apply_cli_overrides(config: Config, args: argparse.Namespace, parser: argparse.ArgumentParser, checkpoints: set[str]) -> None:
    """Override saved config with explicitly passed CLI args."""
    defaults = {a.dest: a.default for a in parser._actions}
    if args.lead_model != defaults.get("lead_model"):
        config.lead_model = args.lead_model
    if args.worker_model != defaults.get("worker_model"):
        config.worker_model = args.worker_model
    if args.lead_effort != defaults.get("lead_effort"):
        config.lead_effort = args.lead_effort
    if args.worker_effort != defaults.get("worker_effort"):
        config.worker_effort = args.worker_effort
    if args.max_budget != defaults.get("max_budget"):
        config.max_budget = args.max_budget
        # Recalculate derived budgets
        config.lead_budget = config.max_budget * 0.2
        config.worker_budget = config.max_budget * 0.1
        config.review_budget = config.max_budget * 0.05
    if args.timeout != defaults.get("timeout"):
        config.timeout = args.timeout
    if args.stall_timeout != defaults.get("stall_timeout"):
        config.stall_timeout = args.stall_timeout
    if args.checkpoint != defaults.get("checkpoint"):
        config.checkpoints = checkpoints


def _collect_all_deps(tasks: dict, target_id: str) -> set[str]:
    """Recursively collect all transitive dependencies of target_id."""
    deps: set[str] = set()
    stack = list(tasks[target_id].depends_on)
    while stack:
        tid = stack.pop()
        if tid not in deps and tid in tasks:
            deps.add(tid)
            stack.extend(tasks[tid].depends_on)
    return deps


def _print_status(ws: Workspace) -> None:
    if not (ws.base / "task_graph.json").exists():
        print("No .nanoteam/ state found.")
        return

    graph = ws.load_task_graph()
    print(f"Goal: {graph.goal}")
    print(f"Tasks: {len(graph.tasks)}")
    print(f"Roles: {', '.join(graph.roles.keys()) or 'none'}")
    print(f"Total cost: ${graph.total_cost:.2f}")
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


def _diagnose(ws: Workspace, args: argparse.Namespace) -> None:
    if not (ws.base / "task_graph.json").exists():
        print("No .nanoteam/ state found.", file=sys.stderr)
        sys.exit(1)

    from .claude import invoke_claude
    from .prompts import lead_diagnose_prompt

    graph = ws.load_task_graph()

    # Build task graph summary
    lines = [f"Goal: {graph.goal}", f"Total cost: ${graph.total_cost:.2f}", ""]
    for task in graph.tasks.values():
        deps = f" (depends: {', '.join(task.depends_on)})" if task.depends_on else ""
        files = f" [changed: {', '.join(task.changed_files)}]" if task.changed_files else ""
        lines.append(f"  {task.id}: {task.title} [{task.status.value}] attempt={task.attempt}{deps}{files}")
    graph_summary = "\n".join(lines)

    # Recent events from log
    events = ws.read_log()
    recent = events[-30:] if len(events) > 30 else events
    import json
    recent_events = "\n".join(json.dumps(e, ensure_ascii=False) for e in recent)

    # Recent turns (last few response files)
    recent_turns = _gather_recent_turns(ws, graph)

    sys_prompt, user_prompt = lead_diagnose_prompt(
        graph.goal, graph_summary, recent_events, recent_turns,
    )

    print("[nanoteam] Running diagnosis...", file=sys.stderr)
    result = invoke_claude(
        user_prompt,
        system_prompt=sys_prompt,
        model=args.lead_model,
        max_budget_usd=1.0,
        cwd=str(ws.root),
    )

    if result.success:
        print(result.output)
    else:
        print(f"Diagnosis failed: {result.error}", file=sys.stderr)
        sys.exit(1)

    print(f"\n[nanoteam] Diagnosis cost: ${result.cost_usd:.2f}", file=sys.stderr)


def _gather_recent_turns(ws: Workspace, graph) -> str:
    """Gather the most recent turn responses for context."""
    parts: list[str] = []
    turns_dir = ws.base / "turns"

    # Global turns (planning, feedback)
    if turns_dir.exists():
        for f in sorted(turns_dir.glob("*-response.md"))[-3:]:
            content = f.read_text()
            if len(content) > 3000:
                content = content[:3000] + "\n... (truncated)"
            parts.append(f"### {f.name}\n{content}")

    # Per-task turns (most recent failed/in-progress tasks first)
    priority_tasks = [
        t for t in graph.tasks.values()
        if t.status in (TaskStatus.FAILED, TaskStatus.IN_PROGRESS, TaskStatus.REVIEW)
    ]
    for task in priority_tasks[:3]:
        task_turns = ws.base / "tasks" / task.id / "turns"
        if task_turns.exists():
            for f in sorted(task_turns.glob("*-response.md"))[-2:]:
                content = f.read_text()
                if len(content) > 3000:
                    content = content[:3000] + "\n... (truncated)"
                parts.append(f"### {task.id}/{f.name}\n{content}")

    return "\n\n".join(parts) if parts else "No turns recorded yet."


if __name__ == "__main__":
    main()
