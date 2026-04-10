from __future__ import annotations

import json
import tempfile
from pathlib import Path

from .models import Role, Task, TaskGraph


class Workspace:
    def __init__(self, root: Path):
        self.root = root
        self.base = root / ".nanoteam"

    def init(self, goal: str) -> None:
        (self.base / "team" / "roles").mkdir(parents=True, exist_ok=True)
        (self.base / "tasks").mkdir(parents=True, exist_ok=True)
        self._write(self.base / "goal.md", goal)
        self._write(self.base / "decisions.md", "# Decisions\n")
        graph = TaskGraph(goal=goal)
        self.save_task_graph(graph)

    def read_goal(self) -> str:
        return (self.base / "goal.md").read_text()

    # -- Task Graph --

    def load_task_graph(self) -> TaskGraph:
        text = (self.base / "task_graph.json").read_text()
        return TaskGraph.from_json(text)

    def save_task_graph(self, graph: TaskGraph) -> None:
        self._write(self.base / "task_graph.json", graph.to_json())

    # -- Tasks --

    def _task_dir(self, task_id: str) -> Path:
        d = self.base / "tasks" / task_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_task_spec(self, task_id: str, spec: str) -> None:
        self._write(self._task_dir(task_id) / "spec.md", spec)

    def read_task_spec(self, task_id: str) -> str:
        return (self._task_dir(task_id) / "spec.md").read_text()

    def write_task_context(self, task_id: str, context: str) -> None:
        self._write(self._task_dir(task_id) / "context.md", context)

    def read_task_context(self, task_id: str) -> str:
        p = self._task_dir(task_id) / "context.md"
        return p.read_text() if p.exists() else ""

    def write_task_result(self, task_id: str, result: str) -> None:
        self._write(self._task_dir(task_id) / "result.md", result)

    def read_task_result(self, task_id: str) -> str | None:
        p = self._task_dir(task_id) / "result.md"
        return p.read_text() if p.exists() else None

    # -- Roles --

    def write_role(self, role: Role) -> None:
        content = f"# {role.name}\n\n{role.description}\n\n"
        content += f"Allowed tools: {', '.join(role.allowed_tools)}\n"
        content += f"Allowed dirs: {', '.join(role.allowed_dirs)}\n"
        self._write(self.base / "team" / "roles" / f"{role.name}.md", content)

    # -- Decisions --

    def append_decision(self, decision: str) -> None:
        p = self.base / "decisions.md"
        current = p.read_text() if p.exists() else "# Decisions\n"
        current += f"\n- {decision}"
        self._write(p, current)

    def read_decisions(self) -> str:
        p = self.base / "decisions.md"
        return p.read_text() if p.exists() else ""

    # -- File snapshots --

    _IGNORE_DIRS = {".nanoteam", ".git", "__pycache__", ".venv", "node_modules", ".mypy_cache"}

    def snapshot_files(self) -> dict[str, float]:
        """Return {relative_path: mtime} for all files in the project root."""
        snapshot: dict[str, float] = {}
        for p in self.root.rglob("*"):
            if p.is_file() and not any(part in self._IGNORE_DIRS for part in p.parts):
                rel = str(p.relative_to(self.root))
                snapshot[rel] = p.stat().st_mtime
        return snapshot

    def diff_files(self, before: dict[str, float], after: dict[str, float]) -> list[str]:
        """Return list of files that were added or modified between two snapshots."""
        changed = []
        for path, mtime in after.items():
            if path not in before or mtime != before[path]:
                changed.append(path)
        return sorted(changed)

    # -- Dynamic context --

    def build_dynamic_context(self, task: Task, graph: TaskGraph) -> str:
        """Build enriched context by combining static context with dependency outputs."""
        parts: list[str] = []

        # Static context from planning phase
        static_ctx = self.read_task_context(task.id)
        if static_ctx:
            parts.append(static_ctx)

        # Dependency task outputs
        dep_parts: list[str] = []
        for dep_id in task.depends_on:
            dep_task = graph.tasks.get(dep_id)
            if not dep_task:
                continue

            dep_section = f"### {dep_id}: {dep_task.title}\n"

            # What the dependency produced (result summary)
            result = self.read_task_result(dep_id)
            if result:
                # Truncate long results to keep context manageable
                if len(result) > 2000:
                    result = result[:2000] + "\n... (truncated)"
                dep_section += f"\n**Result:**\n{result}\n"

            # Which files were changed
            if dep_task.changed_files:
                dep_section += f"\n**Changed files:** {', '.join(dep_task.changed_files)}\n"

                # Inline small files so the worker can see the actual code
                for fpath in dep_task.changed_files:
                    full = self.root / fpath
                    if full.exists() and full.is_file():
                        try:
                            content = full.read_text()
                        except (UnicodeDecodeError, OSError):
                            continue
                        if len(content) <= 5000:
                            dep_section += f"\n**{fpath}:**\n```\n{content}\n```\n"
                        else:
                            dep_section += f"\n**{fpath}:** ({len(content)} chars, too large to inline)\n"

            dep_parts.append(dep_section)

        if dep_parts:
            parts.append("## Prior Work (from dependency tasks)\n\n" + "\n".join(dep_parts))

        return "\n\n".join(parts) if parts else ""

    # -- Atomic write --

    def _write(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with open(fd, "w") as f:
                f.write(content)
            Path(tmp).replace(path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
