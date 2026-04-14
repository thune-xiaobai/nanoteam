from __future__ import annotations

import json
import select
import sys
from dataclasses import dataclass
from pathlib import Path

from .claude import ClaudeResult, invoke_claude, resume_claude
from .models import Role, Task, TaskGraph, TaskStatus
from .prompts import (
    lead_checkpoint_chat_prompt,
    lead_feedback_prompt,
    lead_planning_prompt,
    lead_replan_prompt,
    lead_review_prompt,
    worker_prompt,
    worker_resume_prompt,
)
from .workspace import Workspace


@dataclass
class Config:
    lead_model: str = "claude-opus-4-6"
    worker_model: str = "claude-opus-4-6"
    lead_effort: str = "high"
    worker_effort: str = "medium"
    max_budget: float = 10.0
    lead_budget: float | None = None
    worker_budget: float | None = None
    review_budget: float | None = None
    timeout: int = 3600
    stall_timeout: int = 300
    checkpoints: set[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.checkpoints is None:
            self.checkpoints = {"plan", "finish"}
        # Default per-invocation budgets: fraction of total, no artificial ceiling
        if self.lead_budget is None:
            self.lead_budget = self.max_budget * 0.2
        if self.worker_budget is None:
            self.worker_budget = self.max_budget * 0.1
        if self.review_budget is None:
            self.review_budget = self.max_budget * 0.05

    def to_dict(self) -> dict:
        return {
            "lead_model": self.lead_model,
            "worker_model": self.worker_model,
            "lead_effort": self.lead_effort,
            "worker_effort": self.worker_effort,
            "max_budget": self.max_budget,
            "lead_budget": self.lead_budget,
            "worker_budget": self.worker_budget,
            "review_budget": self.review_budget,
            "timeout": self.timeout,
            "stall_timeout": self.stall_timeout,
            "checkpoints": sorted(self.checkpoints),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Config:
        if "checkpoints" in data:
            data["checkpoints"] = set(data["checkpoints"])
        return cls(**data)


class Orchestrator:
    def __init__(self, workspace: Workspace, config: Config):
        self.ws = workspace
        self.config = config
        self.total_cost = 0.0
        self._turn_counter = 0
        self._role_sessions: dict[str, str] = {}  # role -> last successful session_id

    def run(self) -> None:
        graph = self.ws.load_task_graph()
        self.total_cost = graph.total_cost

        if not graph.tasks:
            graph = self._plan(graph)
            self.ws.save_task_graph(graph)
            if "plan" in self.config.checkpoints:
                self._checkpoint(graph, "plan")
        else:
            # Resume: re-show plan checkpoint if no tasks have completed yet
            done = any(t.status == TaskStatus.DONE for t in graph.tasks.values())
            if not done and "plan" in self.config.checkpoints:
                self._checkpoint(graph, "plan")

            # Resume interrupted tasks via their Claude sessions
            self._resume_interrupted_tasks(graph)

        while not graph.is_complete():
            if self.total_cost >= self.config.max_budget:
                _log(f"Budget exhausted: ${self.total_cost:.2f} >= ${self.config.max_budget:.2f}")
                prev_budget = self.config.max_budget
                self._checkpoint(graph, "budget")
                self.ws.save_task_graph(graph)
                if self.config.max_budget <= prev_budget:
                    break  # user didn't increase budget
                continue

            ready = graph.ready_tasks()
            if not ready:
                in_progress = [t for t in graph.tasks.values() if t.status == TaskStatus.IN_PROGRESS]
                if in_progress:
                    _log("Tasks still in progress, but none ready. Possible stall.")
                else:
                    _log("No tasks ready. Entering interactive recovery...")

                # Snapshot state before checkpoint
                prev_snapshot = {t.id: (t.status, t.attempt) for t in graph.tasks.values()}
                prev_count = len(graph.tasks)
                had_interaction = self._checkpoint(graph, "stuck")
                self.ws.save_task_graph(graph)

                # If user made changes (skip/retry/add/remove), continue the loop
                new_snapshot = {t.id: (t.status, t.attempt) for t in graph.tasks.values()}
                graph_changed = new_snapshot != prev_snapshot or len(graph.tasks) != prev_count
                if graph_changed or had_interaction:
                    continue
                break  # user pressed Enter without any interaction — stop

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

        # Resume previous planning session if available
        if graph.plan_session_id:
            _log(f"Resuming planning session {graph.plan_session_id[:8]}...")
            result = resume_claude(
                graph.plan_session_id,
                "You were interrupted during planning. Please output the complete JSON plan now. "
                "Output ONLY the JSON object matching the schema, with no extra text.",
                model=self.config.lead_model,
                cwd=str(self.ws.root),
                timeout=self.config.timeout,
                stall_timeout=self.config.stall_timeout,
            )
            self.total_cost += result.cost_usd
            graph.total_cost = self.total_cost
            if result.success:
                plan = self._parse_json(result.output)
                if plan.get("tasks"):
                    return self._apply_plan(graph, plan)
            _log("Planning session resume failed, starting fresh...")
            graph.plan_session_id = None

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

        # Save session for potential resume
        if result.session_id:
            graph.plan_session_id = result.session_id
            self.ws.save_task_graph(graph)

        plan = self._parse_json(result.output)

        if not plan.get("tasks") and result.session_id:
            _log("Lead returned no tasks — retrying in same session...")
            retry_prompt = (
                "Your previous response did not contain a valid JSON plan. "
                "Please output ONLY the JSON object matching the schema, with no extra text."
            )
            retry = resume_claude(
                result.session_id, retry_prompt,
                model=self.config.lead_model,
                cwd=str(self.ws.root),
            )
            self.total_cost += retry.cost_usd
            graph.total_cost = self.total_cost
            if retry.success:
                plan = self._parse_json(retry.output)

        if not plan.get("tasks"):
            _log(f"Lead returned no tasks. Raw output: {(result.output or '')[:200]}")
            raise RuntimeError("Planning produced no tasks")

        return self._apply_plan(graph, plan)

    def _apply_plan(self, graph: TaskGraph, plan: dict) -> TaskGraph:
        """Apply a parsed plan dict to the task graph."""
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
                description=rdata.get("description", rdata["name"]),
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

    def _resume_interrupted_tasks(self, graph: TaskGraph) -> None:
        """Resume tasks that were IN_PROGRESS when the previous run was interrupted."""
        interrupted = [t for t in graph.tasks.values() if t.status == TaskStatus.IN_PROGRESS]
        for task in interrupted:
            if not task.session_id:
                _log(f"Task {task.id} has no session_id, resetting to READY")
                task.status = TaskStatus.READY
                self.ws.save_task_graph(graph)
                continue

            _log(f"Resuming {task.id}: {task.title} (session {task.session_id[:8]}...)")
            self.ws.append_log({"event": "task_resume", "task_id": task.id, "session_id": task.session_id})

            before = self.ws.snapshot_files()
            result = resume_claude(
                task.session_id,
                "You were interrupted. Continue and complete your task. "
                "Output your final result when done.",
                model=task.assigned_model,
                cwd=str(self.ws.root),
                timeout=self.config.timeout,
                stall_timeout=self.config.stall_timeout,
            )
            self.total_cost += result.cost_usd
            graph.total_cost = self.total_cost

            after = self.ws.snapshot_files()
            task.changed_files = self.ws.diff_files(before, after)

            if not result.success:
                _log(f"Resume failed for {task.id}: {result.error}, resetting to READY")
                task.status = TaskStatus.READY
                task.session_id = None
            else:
                self.ws.write_task_result(task.id, result.output)
                task.status = TaskStatus.REVIEW
                verdict = self._review(graph, task)
                if verdict.get("verdict") == "accept":
                    task.status = TaskStatus.DONE
                    _log(f"Task {task.id} accepted: {verdict.get('reason', '')[:80]}")
                    self.ws.append_log({"event": "task_done", "task_id": task.id, "verdict": "accept"})
                else:
                    task.status = TaskStatus.FAILED
                    task.attempt += 1
                    _log(f"Task {task.id} rejected after resume: {verdict.get('reason', '')[:80]}")

            self.ws.save_task_graph(graph)

    MAX_AUTO_RETRIES = 3  # Safety cap for auto-retries on stall with activity

    def _execute_task(self, graph: TaskGraph, task: Task) -> None:
        _log(f"Executing {task.id}: {task.title} (role={task.role}, model={task.assigned_model})")
        self.ws.append_log({"event": "task_start", "task_id": task.id, "role": task.role})

        task.status = TaskStatus.IN_PROGRESS
        self.ws.save_task_graph(graph)

        role = graph.roles.get(task.role) if task.role else None
        spec = self.ws.read_task_spec(task.id)
        role_def = role.description if role else "General-purpose software engineer."

        # Auto-retry loop: keep retrying while worker had activity (transient failure)
        # Stop when: success, no activity (task problem), or safety cap reached
        auto_retries = 0
        while True:
            before = self.ws.snapshot_files()
            context = self.ws.build_dynamic_context(task, graph)

            # Try resuming a previous session for this role (avoids re-reading codebase)
            prev_session = self._role_sessions.get(task.role) if task.role else None
            worker_result = None

            if prev_session:
                _log(f"  Resuming session from previous {task.role} task...")
                prompt = worker_resume_prompt(spec, context)
                worker_result = self._resume_and_record(
                    graph, task.id, "execute", prompt, prev_session,
                    model=task.assigned_model,
                    cwd=str(self.ws.root),
                )
                if not worker_result.success:
                    _log(f"  Resume failed, falling back to fresh session")
                    self._role_sessions.pop(task.role, None)
                    worker_result = None

            if worker_result is None:
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

            # Save session_id for potential resume
            if worker_result.session_id:
                task.session_id = worker_result.session_id
                self.ws.save_task_graph(graph)

            # Snapshot after and record changed files
            after = self.ws.snapshot_files()
            task.changed_files = self.ws.diff_files(before, after)
            if task.changed_files:
                _log(f"  Files changed: {', '.join(task.changed_files)}")

            if worker_result.success:
                break

            # Worker failed — decide whether to auto-retry
            if worker_result.had_activity and auto_retries < self.MAX_AUTO_RETRIES:
                auto_retries += 1
                _log(f"Worker had activity but failed ({worker_result.error}), auto-retrying ({auto_retries}/{self.MAX_AUTO_RETRIES})...")
                self.ws.append_log({"event": "auto_retry", "task_id": task.id, "attempt": auto_retries, "error": (worker_result.error or "")[:200]})
                continue

            # No activity or retries exhausted — fall through to review/replan
            _log(f"Worker failed on {task.id}: {worker_result.error}")
            break

        if not worker_result.success:
            self.ws.write_task_result(task.id, f"FAILED: {worker_result.error}")
            if task.role:
                self._role_sessions.pop(task.role, None)
        else:
            self.ws.write_task_result(task.id, worker_result.output)
            if task.role and worker_result.session_id:
                self._role_sessions[task.role] = worker_result.session_id

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

        # Pre-check: if worker claims file changes but none actually happened, auto-reject
        if not task.changed_files and _claims_file_changes(result):
            reason = "Worker claimed file changes but no files were actually modified."
            _log(f"  Auto-rejecting {task.id}: {reason}")
            return {"verdict": "reject", "reason": reason}

        sys_prompt, user_prompt = lead_review_prompt(
            graph.goal, task, spec, result, graph.decisions,
            changed_files=task.changed_files,
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
        """Raise if over budget. Only used outside the main loop (e.g. planning)."""
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

    def _resume_and_record(
        self,
        graph: TaskGraph,
        task_id: str | None,
        phase: str,
        prompt: str,
        session_id: str,
        **kwargs,
    ) -> ClaudeResult:
        """Resume a Claude session with turn recording and logging."""
        self._turn_counter += 1
        turn = self._turn_counter

        self.ws.append_log({
            "event": "invoke_claude",
            "task_id": task_id,
            "phase": phase,
            "model": kwargs.get("model", "?"),
            "turn": turn,
            "resumed_session": session_id[:8],
        })

        result = resume_claude(
            session_id,
            prompt,
            timeout=kwargs.pop("timeout", self.config.timeout),
            stall_timeout=kwargs.pop("stall_timeout", self.config.stall_timeout),
            **kwargs,
        )
        self.total_cost += result.cost_usd
        graph.total_cost = self.total_cost
        self.ws.save_task_graph(graph)

        prompt_path, response_path = self.ws.write_turn(
            task_id, turn, phase,
            prompt, result.output or result.error or "",
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
            "resumed": True,
        })

        return result

    def _parse_json(self, text: str) -> dict:
        text = text.strip()
        # Strip markdown fences
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Extract JSON from mixed text: find first { and last }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        _log(f"JSON parse error. Raw text: {text[:200]}")
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

    def _checkpoint(self, graph: TaskGraph, phase: str) -> bool:
        """Pause for client review and conversational interaction.

        Returns True if the user provided any input during the checkpoint.
        """
        self._print_status(graph, phase)

        # Show task specs at plan checkpoint so user can see the full plan
        if phase == "plan":
            self._print_plan_details(graph)

        print(file=sys.stderr)
        if phase == "stuck":
            failed = [t for t in graph.tasks.values() if t.status == TaskStatus.FAILED]
            if failed:
                _log("Quick commands:")
                _log("  skip <task-id>   — mark a task as done (skip it)")
                _log("  retry <task-id>  — reset a failed task for retry")
                _log("  Or type a question/instruction for Lead")
            _log("Press Enter to stop, or type a command:")
        elif phase == "budget":
            _log(f"Budget ${self.config.max_budget:.2f} exhausted (spent ${self.total_cost:.2f}).")
            _log("  Type a number to set new budget (e.g. '200')")
            _log("  Or press Enter to stop:")
        else:
            _log("Press Enter to continue, or type a question/feedback:")

        had_interaction = False

        # Conversation loop: user can ask questions or give feedback multiple times
        while True:
            try:
                user_input = _read_multiline("> ")
            except EOFError:
                break
            except KeyboardInterrupt:
                print(file=sys.stderr)
                raise

            if not user_input:
                break

            had_interaction = True
            self._handle_checkpoint_input(graph, user_input)
            print(file=sys.stderr)
            _log("Press Enter to continue, or type another question/feedback:")

        return had_interaction

    def _print_status(self, graph: TaskGraph, phase: str) -> None:
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
            for d in graph.decisions[-5:]:
                text = d.get("text", str(d)) if isinstance(d, dict) else d
                _log(f"  - {text[:100]}")

    def _print_plan_details(self, graph: TaskGraph) -> None:
        """Show spec preview for each pending task so user can review the plan."""
        pending = [t for t in graph.tasks.values() if t.status in (TaskStatus.PENDING, TaskStatus.READY)]
        if not pending:
            return

        print(file=sys.stderr)
        _log("─── Plan Details ───")
        for t in pending:
            spec = self.ws.read_task_spec(t.id) or ""
            # Show first 200 chars as preview
            preview = spec[:200].replace("\n", " ")
            if len(spec) > 200:
                preview += "..."
            _log(f"  {t.id} [{t.role or 'unassigned'}]: {preview}")

    def _handle_checkpoint_input(self, graph: TaskGraph, user_input: str) -> None:
        """Handle user input: quick commands first, then delegate to Lead."""
        graph.feedback_log.append(user_input)
        self.ws.append_log({"event": "checkpoint_chat", "input": user_input[:200]})

        # Quick commands — no Lead invocation needed
        if self._handle_quick_command(graph, user_input):
            return

        # Delegate to Lead for complex requests
        task_specs = {}
        for tid in graph.tasks:
            task_specs[tid] = self.ws.read_task_spec(tid) or ""

        sys_prompt, user_prompt = lead_checkpoint_chat_prompt(
            graph.goal, graph, task_specs, user_input,
        )
        result = self._invoke_and_record(
            graph, None, "checkpoint_chat", sys_prompt, user_prompt,
            model=self.config.lead_model,
            effort=self.config.lead_effort,
            max_budget_usd=self.config.review_budget,
            cwd=str(self.ws.root),
        )

        if not result.success:
            _log(f"Lead response failed: {result.error}")
            return

        response = self._parse_json(result.output)
        resp_type = response.get("type", "answer")

        if resp_type == "answer":
            # Just print the answer
            print(file=sys.stderr)
            _log(f"Lead: {response.get('content', '(no response)')}")
        else:
            # Apply plan modifications
            self._apply_plan_changes(graph, response)

    def _handle_quick_command(self, graph: TaskGraph, user_input: str) -> bool:
        """Handle direct commands without invoking Lead. Returns True if handled."""
        parts = user_input.strip().split(None, 1)
        cmd = parts[0].lower() if parts else ""
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd == "skip" and arg:
            if arg not in graph.tasks:
                _log(f"Unknown task: {arg}")
            elif graph.tasks[arg].status == TaskStatus.DONE:
                _log(f"{arg} is already done")
            else:
                graph.tasks[arg].status = TaskStatus.DONE
                self.ws.save_task_graph(graph)
                _log(f"Skipped {arg}")
            return True

        if cmd == "retry" and arg:
            if arg not in graph.tasks:
                _log(f"Unknown task: {arg}")
            elif graph.tasks[arg].status != TaskStatus.FAILED:
                _log(f"{arg} is not failed (status: {graph.tasks[arg].status.value})")
            else:
                graph.tasks[arg].status = TaskStatus.PENDING
                graph.tasks[arg].attempt = 0
                self.ws.save_task_graph(graph)
                _log(f"Reset {arg} for retry")
            return True

        if cmd == "budget" and arg:
            try:
                new_budget = float(arg)
                if new_budget > self.total_cost:
                    self.config.max_budget = new_budget
                    _log(f"Budget increased to ${new_budget:.2f}")
                else:
                    _log(f"New budget must be > current cost ${self.total_cost:.2f}")
            except ValueError:
                _log(f"Invalid budget: {arg}")
            return True

        # Plain number → budget shorthand
        if user_input.strip().replace(".", "", 1).isdigit():
            try:
                new_budget = float(user_input.strip())
                if new_budget > self.total_cost:
                    self.config.max_budget = new_budget
                    _log(f"Budget increased to ${new_budget:.2f}")
                else:
                    _log(f"New budget must be > current cost ${self.total_cost:.2f}")
            except ValueError:
                return False
            return True

        return False

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

        self._apply_plan_changes(graph, self._parse_json(result.output))

    def _apply_plan_changes(self, graph: TaskGraph, changes: dict) -> None:
        """Apply plan modifications from Lead response."""
        # Modify existing tasks (now also handles FAILED tasks)
        for mod in changes.get("modify_tasks", []):
            tid = mod.get("id")
            if tid and tid in graph.tasks:
                task = graph.tasks[tid]
                if task.status in (TaskStatus.PENDING, TaskStatus.READY, TaskStatus.FAILED):
                    if mod.get("updated_spec"):
                        self.ws.write_task_spec(tid, mod["updated_spec"])
                        _log(f"  Updated spec for {tid}")
                    if mod.get("updated_title"):
                        task.title = mod["updated_title"]
                        _log(f"  Updated title for {tid}: {task.title}")
                    # Reset failed tasks so they can be retried
                    if task.status == TaskStatus.FAILED:
                        task.status = TaskStatus.PENDING
                        task.attempt = 0
                        _log(f"  Reset {tid} for retry")

        # Retry failed tasks (reset to PENDING)
        for tid in changes.get("retry_tasks", []):
            if tid in graph.tasks and graph.tasks[tid].status == TaskStatus.FAILED:
                graph.tasks[tid].status = TaskStatus.PENDING
                graph.tasks[tid].attempt = 0
                _log(f"  Retrying task {tid}")

        # Skip tasks (mark as DONE)
        for tid in changes.get("skip_tasks", []):
            if tid in graph.tasks and graph.tasks[tid].status in (
                TaskStatus.FAILED, TaskStatus.PENDING, TaskStatus.READY,
            ):
                graph.tasks[tid].status = TaskStatus.DONE
                _log(f"  Skipped task {tid}")

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
                description=rdata.get("description", rdata["name"]),
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

        content = changes.get("content", "")
        if content:
            _log(f"Lead: {content}")
        _log("Plan updated.")


def _log(msg: str) -> None:
    print(f"[nanoteam] {msg}", file=sys.stderr)


def _read_multiline(prompt: str = "> ") -> str:
    """Read input, accumulating multi-line paste automatically.

    Typed input works as before (single line, Enter submits).
    Pasted multi-line text is detected via stdin buffering and
    accumulated into one string.  When a multi-line paste is
    detected, the user is shown a summary and can add context
    before confirming.
    """
    first_line = input(prompt)
    lines = [first_line]

    # After the first line, drain any remaining lines in the buffer
    # (indicates a multi-line paste rather than typed input)
    while select.select([sys.stdin], [], [], 0.05)[0]:
        line = sys.stdin.readline()
        if not line:  # EOF
            break
        lines.append(line.rstrip("\n"))

    text = "\n".join(lines).strip()

    if len(lines) > 1:
        _log(f"(Pasted {len(lines)} lines)")
        extra = input("  Add context or Enter to send: ").strip()
        if extra:
            text = text + "\n\n" + extra

    return text


import re

_FILE_CHANGE_PATTERN = re.compile(
    r'\b(?:creat|writ|wrote|generat|added|built)\w*\s+.*?'
    r'(?:\.\w{1,5}\b|/\w+)',
    re.IGNORECASE,
)


def _claims_file_changes(result: str) -> bool:
    """Heuristic: does the result text claim to have created/written files?"""
    return bool(_FILE_CHANGE_PATTERN.search(result))
