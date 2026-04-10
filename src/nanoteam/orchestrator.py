from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from .claude import ClaudeResult, invoke_claude
from .models import Role, Task, TaskGraph, TaskStatus
from .prompts import (
    lead_feedback_prompt,
    lead_planning_prompt,
    lead_replan_prompt,
    lead_review_prompt,
    worker_prompt,
)
from .workspace import Workspace


@dataclass
class Config:
    lead_model: str = "claude-opus-4-6"
    worker_model: str = "claude-opus-4-6"
    lead_effort: str = "high"
    worker_effort: str = "medium"
    max_budget: float = 10.0
    lead_budget: float = 2.0
    worker_budget: float = 1.0
    review_budget: float = 0.5
    timeout: int = 3600
    stall_timeout: int = 300
    checkpoints: set[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.checkpoints is None:
            self.checkpoints = {"plan", "finish"}


class Orchestrator:
    def __init__(self, workspace: Workspace, config: Config):
        self.ws = workspace
        self.config = config
        self.total_cost = 0.0
        self._turn_counter = 0

    def run(self) -> None:
        graph = self.ws.load_task_graph()
        self.total_cost = graph.total_cost

        if not graph.tasks:
            graph = self._plan(graph)
            self.ws.save_task_graph(graph)
            if "plan" in self.config.checkpoints:
                self._checkpoint(graph, "plan")

        while not graph.is_complete():
            self._check_budget()

            ready = graph.ready_tasks()
            if not ready:
                in_progress = [t for t in graph.tasks.values() if t.status == TaskStatus.IN_PROGRESS]
                if in_progress:
                    _log("Tasks still in progress, but none ready. Possible stall.")
                else:
                    _log("No tasks ready and none in progress. Deadlock detected.")
                break

            for task in ready:
                self._execute_task(graph, task)
                self.ws.save_task_graph(graph)

            if "phase" in self.config.checkpoints and not graph.is_complete():
                self._checkpoint(graph, "phase")

        if graph.is_complete():
            self._finalize(graph)
            if "finish" in self.config.checkpoints:
                self._checkpoint(graph, "finish")
        else:
            _log(f"Stopped. {sum(1 for t in graph.tasks.values() if t.status == TaskStatus.DONE)}/{len(graph.tasks)} tasks done.")
            _log(f"Total cost: ${self.total_cost:.2f}")

    def _plan(self, graph: TaskGraph) -> TaskGraph:
        _log("Planning: asking Lead to decompose the goal...")
        self.ws.append_log({"event": "plan_start", "goal": graph.goal})

        sys_prompt, user_prompt = lead_planning_prompt(graph.goal)
        result = self._invoke_and_record(
            graph, None, "plan", sys_prompt, user_prompt,
            model=self.config.lead_model,
            effort=self.config.lead_effort,
            max_budget_usd=self.config.lead_budget,
            cwd=str(self.ws.root),
        )

        if not result.success:
            _log(f"Lead planning failed: {result.error}")
            raise RuntimeError(f"Planning failed: {result.error}")

        plan = self._parse_json(result.output)

        for tdata in plan.get("tasks", []):
            task = Task(
                id=tdata["id"],
                title=tdata["title"],
                depends_on=tdata.get("depends_on", []),
                role=tdata.get("role"),
                assigned_model=self.config.worker_model,
            )
            graph.tasks[task.id] = task
            self.ws.write_task_spec(task.id, tdata.get("spec", task.title))
            if tdata.get("context"):
                ctx = tdata["context"]
                if isinstance(ctx, list):
                    ctx = "\n".join(str(c) for c in ctx)
                self.ws.write_task_context(task.id, str(ctx))

        for rdata in plan.get("roles", []):
            role = Role(
                name=rdata["name"],
                description=rdata["description"],
                allowed_tools=rdata.get("allowed_tools", ["Read", "Grep", "Glob", "Edit", "Write", "Bash"]),
                allowed_dirs=rdata.get("allowed_dirs", []),
            )
            graph.roles[role.name] = role
            self.ws.write_role(role)

        graph.decisions.extend(plan.get("decisions", []))
        for d in plan.get("decisions", []):
            self.ws.append_decision(d)

        _log(f"Plan created: {len(graph.tasks)} tasks, {len(graph.roles)} roles")
        self.ws.append_log({"event": "plan_done", "tasks": len(graph.tasks), "roles": len(graph.roles), "cost": self.total_cost})
        return graph

    def _execute_task(self, graph: TaskGraph, task: Task) -> None:
        _log(f"Executing {task.id}: {task.title} (role={task.role}, model={task.assigned_model})")
        self.ws.append_log({"event": "task_start", "task_id": task.id, "role": task.role})

        task.status = TaskStatus.IN_PROGRESS
        self.ws.save_task_graph(graph)

        # Snapshot files before worker runs
        before = self.ws.snapshot_files()

        role = graph.roles.get(task.role) if task.role else None
        spec = self.ws.read_task_spec(task.id)
        context = self.ws.build_dynamic_context(task, graph)
        role_def = role.description if role else "General-purpose software engineer."

        sys_prompt, user_prompt = worker_prompt(spec, context, role_def)

        worker_result = self._invoke_and_record(
            graph, task.id, "execute", sys_prompt, user_prompt,
            model=task.assigned_model,
            effort=self.config.worker_effort,
            allowed_tools=role.allowed_tools if role else None,
            add_dirs=self._resolve_dirs(role) if role else None,
            max_budget_usd=self.config.worker_budget,
            cwd=str(self.ws.root),
        )

        # Snapshot after and record changed files
        after = self.ws.snapshot_files()
        task.changed_files = self.ws.diff_files(before, after)
        if task.changed_files:
            _log(f"  Files changed: {', '.join(task.changed_files)}")

        if not worker_result.success:
            _log(f"Worker failed on {task.id}: {worker_result.error}")
            self.ws.write_task_result(task.id, f"FAILED: {worker_result.error}")
        else:
            self.ws.write_task_result(task.id, worker_result.output)

        task.status = TaskStatus.REVIEW
        verdict = self._review(graph, task)

        if verdict.get("verdict") == "accept":
            task.status = TaskStatus.DONE
            _log(f"Task {task.id} accepted: {verdict.get('reason', '')[:80]}")
            self.ws.append_log({"event": "task_done", "task_id": task.id, "verdict": "accept"})
            if verdict.get("decision"):
                graph.decisions.append(verdict["decision"])
                self.ws.append_decision(verdict["decision"])
        else:
            task.status = TaskStatus.FAILED
            task.attempt += 1
            _log(f"Task {task.id} rejected (attempt {task.attempt}/{task.max_attempts}): {verdict.get('reason', '')[:80]}")
            self.ws.append_log({"event": "task_rejected", "task_id": task.id, "attempt": task.attempt, "reason": verdict.get("reason", "")[:200]})
            if task.attempt < task.max_attempts:
                self._replan_task(graph, task, verdict.get("reason", "Unknown"))
            else:
                _log(f"Task {task.id} exceeded max attempts. Marking as failed.")

    def _review(self, graph: TaskGraph, task: Task) -> dict:
        _log(f"Reviewing {task.id}...")

        spec = self.ws.read_task_spec(task.id)
        result = self.ws.read_task_result(task.id) or "No result."

        sys_prompt, user_prompt = lead_review_prompt(
            graph.goal, task, spec, result, graph.decisions,
        )
        review = self._invoke_and_record(
            graph, task.id, "review", sys_prompt, user_prompt,
            model=self.config.lead_model,
            effort=self.config.lead_effort,
            max_budget_usd=self.config.review_budget,
            cwd=str(self.ws.root),
        )

        if not review.success:
            _log(f"Review failed, auto-accepting: {review.error}")
            return {"verdict": "accept", "reason": "Review invocation failed, auto-accepting."}

        return self._parse_json(review.output)

    def _replan_task(self, graph: TaskGraph, task: Task, reason: str) -> None:
        _log(f"Replanning {task.id}...")

        sys_prompt, user_prompt = lead_replan_prompt(graph.goal, graph, task, reason)
        result = self._invoke_and_record(
            graph, task.id, "replan", sys_prompt, user_prompt,
            model=self.config.lead_model,
            effort=self.config.lead_effort,
            max_budget_usd=self.config.review_budget,
            cwd=str(self.ws.root),
        )

        if not result.success:
            _log(f"Replan failed: {result.error}. Retrying with same spec.")
            task.status = TaskStatus.READY
            return

        plan = self._parse_json(result.output)
        action = plan.get("action", "retry")

        if action == "retry" and plan.get("updated_spec"):
            self.ws.write_task_spec(task.id, plan["updated_spec"])
            task.status = TaskStatus.READY
        elif action == "split":
            for tdata in plan.get("new_tasks", []):
                new_task = Task(
                    id=tdata["id"],
                    title=tdata["title"],
                    depends_on=tdata.get("depends_on", []),
                    role=tdata.get("role", task.role),
                    assigned_model=self.config.worker_model,
                )
                graph.tasks[new_task.id] = new_task
                self.ws.write_task_spec(new_task.id, tdata.get("spec", new_task.title))
                if tdata.get("context"):
                    ctx = tdata["context"]
                    if isinstance(ctx, list):
                        ctx = "\n".join(str(c) for c in ctx)
                    self.ws.write_task_context(new_task.id, str(ctx))
            task.status = TaskStatus.DONE
            _log(f"Task {task.id} split into {len(plan.get('new_tasks', []))} sub-tasks")
        elif action == "reassign" and plan.get("new_role"):
            task.role = plan["new_role"]
            task.status = TaskStatus.READY
        else:
            task.status = TaskStatus.READY

        if plan.get("decision"):
            graph.decisions.append(plan["decision"])
            self.ws.append_decision(plan["decision"])

    def _resolve_dirs(self, role: Role) -> list[str] | None:
        if not role.allowed_dirs:
            return None
        return [str(self.ws.root / d) for d in role.allowed_dirs]

    def _check_budget(self) -> None:
        if self.total_cost >= self.config.max_budget:
            raise RuntimeError(
                f"Budget exhausted: ${self.total_cost:.2f} >= ${self.config.max_budget:.2f}"
            )

    def _invoke_and_record(
        self,
        graph: TaskGraph,
        task_id: str | None,
        phase: str,
        sys_prompt: str,
        user_prompt: str,
        **kwargs,
    ) -> ClaudeResult:
        """Invoke Claude with full turn recording and logging."""
        self._turn_counter += 1
        turn = self._turn_counter

        self.ws.append_log({
            "event": "invoke_claude",
            "task_id": task_id,
            "phase": phase,
            "model": kwargs.get("model", "?"),
            "turn": turn,
        })

        result = invoke_claude(
            user_prompt,
            system_prompt=sys_prompt,
            timeout=kwargs.pop("timeout", self.config.timeout),
            stall_timeout=kwargs.pop("stall_timeout", self.config.stall_timeout),
            **kwargs,
        )
        self.total_cost += result.cost_usd
        graph.total_cost = self.total_cost
        self.ws.save_task_graph(graph)

        prompt_path, response_path = self.ws.write_turn(
            task_id, turn, phase,
            user_prompt, result.output or result.error or "",
            system_prompt=sys_prompt,
        )

        self.ws.append_log({
            "event": "invoke_result",
            "task_id": task_id,
            "phase": phase,
            "success": result.success,
            "cost": result.cost_usd,
            "duration_s": result.duration_s,
            "prompt_file": prompt_path,
            "response_file": response_path,
        })

        return result

    def _parse_json(self, text: str) -> dict:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            _log(f"JSON parse error: {e}. Raw text: {text[:200]}")
            return {"verdict": "accept", "reason": "JSON parse failed, auto-accepting.", "action": "retry"}

    def _finalize(self, graph: TaskGraph) -> None:
        _log("=" * 50)
        _log("All tasks completed!")
        _log(f"Total tasks: {len(graph.tasks)}")
        _log(f"Total cost: ${self.total_cost:.2f}")
        _log(f"Decisions made: {len(graph.decisions)}")
        for d in graph.decisions:
            _log(f"  - {d}")
        _log("=" * 50)

    # -- Checkpoint (client interaction) --

    def _checkpoint(self, graph: TaskGraph, phase: str) -> None:
        """Pause for client review and feedback."""
        done = [t for t in graph.tasks.values() if t.status == TaskStatus.DONE]
        pending = [t for t in graph.tasks.values() if t.status in (TaskStatus.PENDING, TaskStatus.READY)]
        failed = [t for t in graph.tasks.values() if t.status == TaskStatus.FAILED]

        print(file=sys.stderr)
        _log(f"{'═' * 20} CHECKPOINT ({phase}) {'═' * 20}")
        _log(f"Goal: {graph.goal}")
        _log(f"Cost so far: ${self.total_cost:.2f} / ${self.config.max_budget:.2f}")
        print(file=sys.stderr)

        if done:
            _log(f"Completed ({len(done)}):")
            for t in done:
                _log(f"  [+] {t.id}: {t.title}")
        if pending:
            _log(f"Pending ({len(pending)}):")
            for t in pending:
                deps = f" ← depends on {', '.join(t.depends_on)}" if t.depends_on else ""
                _log(f"  [ ] {t.id}: {t.title}{deps}")
        if failed:
            _log(f"Failed ({len(failed)}):")
            for t in failed:
                _log(f"  [x] {t.id}: {t.title} (attempt {t.attempt}/{t.max_attempts})")

        if graph.decisions:
            print(file=sys.stderr)
            _log("Key decisions:")
            for d in graph.decisions[-5:]:  # show last 5
                _log(f"  - {d[:100]}")

        print(file=sys.stderr)
        _log("Press Enter to continue, or type feedback/new requirements:")

        try:
            feedback = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            feedback = ""

        if feedback:
            self._apply_feedback(graph, feedback)

    def _apply_feedback(self, graph: TaskGraph, feedback: str) -> None:
        """Send client feedback to Lead and apply the resulting plan changes."""
        _log("Sending feedback to Lead...")
        graph.feedback_log.append(feedback)
        self.ws.append_log({"event": "feedback", "input": feedback[:200]})

        sys_prompt, user_prompt = lead_feedback_prompt(graph.goal, graph, feedback)
        result = self._invoke_and_record(
            graph, None, "feedback", sys_prompt, user_prompt,
            model=self.config.lead_model,
            effort=self.config.lead_effort,
            max_budget_usd=self.config.lead_budget,
            cwd=str(self.ws.root),
        )

        if not result.success:
            _log(f"Lead feedback processing failed: {result.error}")
            return

        changes = self._parse_json(result.output)

        # Modify existing tasks
        for mod in changes.get("modify_tasks", []):
            tid = mod.get("id")
            if tid and tid in graph.tasks:
                task = graph.tasks[tid]
                if task.status in (TaskStatus.PENDING, TaskStatus.READY):
                    if mod.get("updated_spec"):
                        self.ws.write_task_spec(tid, mod["updated_spec"])
                        _log(f"  Updated spec for {tid}")
                    if mod.get("updated_title"):
                        task.title = mod["updated_title"]
                        _log(f"  Updated title for {tid}: {task.title}")

        # Add new tasks
        for tdata in changes.get("add_tasks", []):
            task = Task(
                id=tdata["id"],
                title=tdata["title"],
                depends_on=tdata.get("depends_on", []),
                role=tdata.get("role"),
                assigned_model=self.config.worker_model,
            )
            graph.tasks[task.id] = task
            self.ws.write_task_spec(task.id, tdata.get("spec", task.title))
            if tdata.get("context"):
                ctx = tdata["context"]
                if isinstance(ctx, list):
                    ctx = "\n".join(str(c) for c in ctx)
                self.ws.write_task_context(task.id, str(ctx))
            _log(f"  Added task {task.id}: {task.title}")

        # Remove pending tasks
        for tid in changes.get("remove_tasks", []):
            if tid in graph.tasks and graph.tasks[tid].status in (TaskStatus.PENDING, TaskStatus.READY):
                del graph.tasks[tid]
                _log(f"  Removed task {tid}")

        # Add new roles
        for rdata in changes.get("add_roles", []):
            role = Role(
                name=rdata["name"],
                description=rdata["description"],
                allowed_tools=rdata.get("allowed_tools", ["Read", "Grep", "Glob", "Edit", "Write", "Bash"]),
                allowed_dirs=rdata.get("allowed_dirs", []),
            )
            graph.roles[role.name] = role
            self.ws.write_role(role)
            _log(f"  Added role {role.name}")

        # Record decisions
        for d in changes.get("decisions", []):
            graph.decisions.append(d)
            self.ws.append_decision(d)

        self.ws.save_task_graph(graph)
        _log("Plan updated based on feedback.")


def _log(msg: str) -> None:
    print(f"[nanoteam] {msg}", file=sys.stderr)
