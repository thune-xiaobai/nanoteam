from __future__ import annotations

from .models import Task, TaskGraph

# -- Lead: Initial Planning --

LEAD_SYSTEM = """\
You are the Lead architect of a software team. You make high-level decisions, \
decompose goals into tasks, define team roles, and review work quality.

You NEVER write code yourself. You plan, delegate, and review.

Always respond with valid JSON matching the requested schema. No markdown fences, no extra text.\
"""


def lead_planning_prompt(goal: str) -> tuple[str, str]:
    user = f"""\
## Objective

{goal}

## Your Job

Analyze this objective and produce a development plan.

1. Break it into concrete, implementable tasks (3-10 tasks).
2. Define roles needed (each role has a name, description, allowed_tools, allowed_dirs).
3. Specify task dependencies (which tasks must finish before others can start).
4. Record key architectural decisions.

## Output Schema

```json
{{
  "tasks": [
    {{
      "id": "task-001",
      "title": "Short description",
      "depends_on": [],
      "role": "role-name",
      "spec": "Detailed task specification with acceptance criteria",
      "context": "List of relevant files/directories the worker needs to know about"
    }}
  ],
  "roles": [
    {{
      "name": "role-name",
      "description": "What this role does and its expertise",
      "allowed_tools": ["Read", "Grep", "Glob", "Edit", "Write", "Bash"],
      "allowed_dirs": ["src/"]
    }}
  ],
  "decisions": [
    "Decision 1: We chose X because Y"
  ]
}}
```

Respond with ONLY the JSON object.\
"""
    return LEAD_SYSTEM, user


# -- Lead: Review --

def lead_review_prompt(
    goal: str,
    task: Task,
    spec: str,
    result: str,
    decisions: list[str],
) -> tuple[str, str]:
    decisions_text = "\n".join(f"- {d}" for d in decisions) if decisions else "None yet."

    user = f"""\
## Project Goal

{goal}

## Architectural Decisions So Far

{decisions_text}

## Task Under Review

- ID: {task.id}
- Title: {task.title}
- Attempt: {task.attempt + 1}/{task.max_attempts}

## Task Specification

{spec}

## Worker's Result

{result}

## Your Job

Review the worker's result against the specification and acceptance criteria.

## Output Schema

```json
{{
  "verdict": "accept" or "reject",
  "reason": "Why you accept or reject",
  "decision": "Any architectural decision to record (null if none)"
}}
```

Respond with ONLY the JSON object.\
"""
    return LEAD_SYSTEM, user


# -- Lead: Replan --

def lead_replan_prompt(
    goal: str,
    graph: TaskGraph,
    failed_task: Task,
    failure_reason: str,
) -> tuple[str, str]:
    tasks_summary = "\n".join(
        f"- {t.id}: {t.title} [{t.status.value}]"
        for t in graph.tasks.values()
    )

    user = f"""\
## Project Goal

{goal}

## Current Task Graph

{tasks_summary}

## Failed Task

- ID: {failed_task.id}
- Title: {failed_task.title}
- Attempt: {failed_task.attempt}/{failed_task.max_attempts}
- Failure reason: {failure_reason}

## Your Job

Decide how to handle this failure. Options:
1. Retry with an updated spec (modify the task)
2. Split into smaller sub-tasks
3. Change the role assignment
4. Abort this task and adjust the plan

## Output Schema

```json
{{
  "action": "retry" or "split" or "reassign" or "abort",
  "updated_spec": "New spec if retrying (null otherwise)",
  "new_tasks": [
    {{"id": "task-XXX", "title": "...", "depends_on": [], "role": "...", "spec": "...", "context": "..."}}
  ],
  "new_role": "role-name if reassigning (null otherwise)",
  "decision": "What you decided and why"
}}
```

Respond with ONLY the JSON object.\
"""
    return LEAD_SYSTEM, user


# -- Worker --

def worker_prompt(
    spec: str,
    context: str,
    role_definition: str,
) -> tuple[str, str]:
    system = f"""\
You are a focused software engineer with the following role:

{role_definition}

Rules:
- Complete ONLY the task described below. Do not work on anything else.
- Write clean, working code. Run tests if the spec requires it.
- Pay close attention to the Prior Work section — it shows what dependency tasks have already done and what files they created/modified. Build on their work, don't redo it.
- When done, summarize what you did and the outcome.\
"""

    user = f"""\
## Task Specification

{spec}

## Context

{context}

## Instructions

1. Read any relevant existing code mentioned in the context and prior work.
2. Implement the task according to the specification, building on what prior tasks have produced.
3. If the spec includes acceptance criteria, verify them.
4. Summarize what you did and the result.\
"""
    return system, user
