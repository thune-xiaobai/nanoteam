from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass

DEFAULT_TIMEOUT = 600  # 10 minutes


@dataclass
class ClaudeResult:
    success: bool
    output: str
    cost_usd: float
    duration_s: float = 0
    error: str | None = None


def invoke_claude(
    prompt: str,
    *,
    system_prompt: str | None = None,
    model: str = "sonnet",
    allowed_tools: list[str] | None = None,
    add_dirs: list[str] | None = None,
    max_budget_usd: float | None = None,
    cwd: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> ClaudeResult:
    cmd = [
        "claude",
        "-p",
        prompt,
        "--output-format", "stream-json",
        "--verbose",
        "--model", model,
        "--permission-mode", "bypassPermissions",
    ]

    if system_prompt:
        cmd.extend(["--system-prompt", system_prompt])

    if allowed_tools:
        cmd.extend(["--allowedTools", " ".join(allowed_tools)])

    if add_dirs:
        for d in add_dirs:
            cmd.extend(["--add-dir", d])

    if max_budget_usd is not None:
        cmd.extend(["--max-budget-usd", str(max_budget_usd)])

    _log(f"Invoking claude ({model}): {prompt[:80]}...")

    start = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
    )

    result_data: dict | None = None
    last_activity = time.monotonic()

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue

            last_activity = time.monotonic()

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            etype = event.get("type", "")

            if etype == "assistant":
                _process_assistant_event(event)
            elif etype == "result":
                result_data = event

            # Check timeout
            elapsed = time.monotonic() - start
            if elapsed >= timeout:
                proc.kill()
                proc.wait()
                _log(f"  TIMEOUT after {elapsed:.0f}s")
                return ClaudeResult(
                    success=False, output="", cost_usd=0,
                    duration_s=elapsed,
                    error=f"Timeout after {elapsed:.0f}s",
                )

        proc.wait()

    except KeyboardInterrupt:
        proc.kill()
        proc.wait()
        raise

    duration = time.monotonic() - start

    if result_data:
        result_text = result_data.get("result", "")
        is_error = result_data.get("is_error", False)
        cost = result_data.get("total_cost_usd", 0) or 0
        num_turns = result_data.get("num_turns", 0)
        _log(f"  done in {duration:.0f}s | cost=${cost:.4f} | turns={num_turns}")
        return ClaudeResult(
            success=not is_error,
            output=result_text,
            cost_usd=cost,
            duration_s=duration,
            error=result_text if is_error else None,
        )

    # Fallback: no result event found
    stderr = proc.stderr.read() if proc.stderr else ""
    _log(f"  finished in {duration:.0f}s but no result event (exit={proc.returncode})")
    return ClaudeResult(
        success=False, output="", cost_usd=0,
        duration_s=duration,
        error=stderr or f"No result event, exit code {proc.returncode}",
    )


def _process_assistant_event(event: dict) -> None:
    """Extract and log what the agent is doing from an assistant message event."""
    msg = event.get("message", {})
    content = msg.get("content", [])

    for block in content:
        btype = block.get("type", "")

        if btype == "tool_use":
            name = block.get("name", "?")
            inp = block.get("input", {})
            detail = _summarize_tool_use(name, inp)
            _log(f"  > {name}: {detail}")

        elif btype == "text":
            text = block.get("text", "").strip()
            if text:
                # Show first line, truncated
                first_line = text.split("\n")[0][:120]
                _log(f"  > reply: {first_line}")


def _summarize_tool_use(name: str, inp: dict) -> str:
    """Produce a short human-readable summary of a tool call."""
    if name == "Read":
        return inp.get("file_path", "?")
    elif name == "Write":
        path = inp.get("file_path", "?")
        content = inp.get("content", "")
        return f"{path} ({len(content)} chars)"
    elif name == "Edit":
        return inp.get("file_path", "?")
    elif name == "Bash":
        cmd = inp.get("command", "?")
        return cmd[:120]
    elif name == "Grep":
        pattern = inp.get("pattern", "?")
        path = inp.get("path", ".")
        return f"'{pattern}' in {path}"
    elif name == "Glob":
        return inp.get("pattern", "?")
    else:
        # Generic: show first key=value
        for k, v in inp.items():
            return f"{k}={str(v)[:80]}"
        return ""


def _log(msg: str) -> None:
    print(f"[nanoteam] {msg}", file=sys.stderr, flush=True)
