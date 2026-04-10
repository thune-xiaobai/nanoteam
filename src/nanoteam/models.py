from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Task:
    id: str
    title: str
    status: TaskStatus = TaskStatus.PENDING
    depends_on: list[str] = field(default_factory=list)
    role: str | None = None
    assigned_model: str = "sonnet"
    attempt: int = 0
    max_attempts: int = 3
    changed_files: list[str] = field(default_factory=list)


@dataclass
class Role:
    name: str
    description: str
    allowed_tools: list[str] = field(default_factory=lambda: ["Read", "Grep", "Glob", "Edit", "Write", "Bash"])
    allowed_dirs: list[str] = field(default_factory=list)


@dataclass
class TaskGraph:
    goal: str
    tasks: dict[str, Task] = field(default_factory=dict)
    roles: dict[str, Role] = field(default_factory=dict)
    decisions: list[str] = field(default_factory=list)
    feedback_log: list[str] = field(default_factory=list)
    total_cost: float = 0.0

    def ready_tasks(self) -> list[Task]:
        ready = []
        for task in self.tasks.values():
            if task.status == TaskStatus.READY:
                ready.append(task)
            elif task.status == TaskStatus.PENDING:
                if all(self.tasks[dep].status == TaskStatus.DONE for dep in task.depends_on):
                    task.status = TaskStatus.READY
                    ready.append(task)
        return ready

    def is_complete(self) -> bool:
        return all(t.status == TaskStatus.DONE for t in self.tasks.values())

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskGraph:
        graph = cls(goal=data["goal"])
        for tid, tdata in data.get("tasks", {}).items():
            tdata["status"] = TaskStatus(tdata["status"])
            graph.tasks[tid] = Task(**tdata)
        for rname, rdata in data.get("roles", {}).items():
            graph.roles[rname] = Role(**rdata)
        graph.decisions = data.get("decisions", [])
        graph.feedback_log = data.get("feedback_log", [])
        graph.total_cost = data.get("total_cost", 0.0)
        return graph

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> TaskGraph:
        return cls.from_dict(json.loads(text))
