from __future__ import annotations

import json
import select
import subprocess
import sys
import time
from dataclasses import dataclass

DEFAULT_TIMEOUT = 3600  # 60 minutes wall clock
DEFAULT_STALL_TIMEOUT = 300  # 5 minutes without output = stalled


@dataclass
class ClaudeResult:
    success: bool
    output: str
    cost_usd: float
    duration_s: float = 0
    error: str | None = None
    session_id: str | None = None
    had_activity: bool = False


def invoke_claude(
    prompt: str,
    *,
    system_prompt: str | None = None,
    model: str = "claude-opus-4-6",
    effort: str | None = None,
    allowed_tools: list[str] | None = None,
    add_dirs: list[str] | None = None,
    max_budget_usd: float | None = None,
    cwd: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    stall_timeout: int = DEFAULT_STALL_TIMEOUT,
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

    if effort:
        cmd.extend(["--effort", effort])

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
        text=False,  # raw bytes for non-blocking reads
        cwd=cwd,
    )

    result_data: dict | None = None
    last_activity = time.monotonic()
    activity_count = 0
    buf = b""

    try:
        assert proc.stdout is not None
        fd = proc.stdout.fileno()

        while True:
            # Use select to wait for data with a short poll interval
            ready, _, _ = select.select([fd], [], [], 5.0)

            now = time.monotonic()
            elapsed = now - start
            idle = now - last_activity

            if elapsed >= timeout:
                proc.kill()
                proc.wait()
                _log(f"  TIMEOUT after {elapsed:.0f}s (wall clock)")
                return ClaudeResult(
                    success=False, output="", cost_usd=0,
                    duration_s=elapsed,
                    error=f"Timeout after {elapsed:.0f}s",
                    had_activity=activity_count > 0,
                )

            if idle >= stall_timeout:
                proc.kill()
                proc.wait()
                _log(f"  STALLED: no output for {idle:.0f}s (elapsed {elapsed:.0f}s, activity={activity_count})")
                return ClaudeResult(
                    success=False, output="", cost_usd=0,
                    duration_s=elapsed,
                    error=f"Stalled: no output for {idle:.0f}s",
                    had_activity=activity_count > 0,
                )

            if not ready:
                continue

            chunk = proc.stdout.read1(8192) if hasattr(proc.stdout, 'read1') else proc.stdout.read(8192)
            if not chunk:
                # EOF — process finished writing
                break

            last_activity = time.monotonic()
            buf += chunk

            # Process complete lines
            while b"\n" in buf:
                line_bytes, buf = buf.split(b"\n", 1)
                line = line_bytes.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                etype = event.get("type", "")

                if etype == "assistant":
                    _process_assistant_event(event)
                    activity_count += 1
                elif etype == "result":
                    result_data = event

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
        session_id = result_data.get("session_id")
        _log(f"  done in {duration:.0f}s | cost=${cost:.4f} | turns={num_turns}")

        # If error with empty result, try to get more info from stderr or the event itself
        error_msg = None
        if is_error:
            error_msg = result_text
            if not error_msg:
                stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
                error_msg = stderr.strip() or f"Unknown error (is_error=true, cost=${cost:.4f}, turns={num_turns})"
                _log(f"  error (empty result): {error_msg[:200]}")

        return ClaudeResult(
            success=not is_error,
            output=result_text,
            cost_usd=cost,
            duration_s=duration,
            error=error_msg,
            session_id=session_id,
            had_activity=activity_count > 0,
        )

    # Fallback: no result event found
    stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
    _log(f"  finished in {duration:.0f}s but no result event (exit={proc.returncode})")
    return ClaudeResult(
        success=False, output="", cost_usd=0,
        duration_s=duration,
        error=stderr or f"No result event, exit code {proc.returncode}",
        had_activity=activity_count > 0,
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


def resume_claude(
    session_id: str,
    prompt: str,
    *,
    model: str = "claude-opus-4-6",
    cwd: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    stall_timeout: int = DEFAULT_STALL_TIMEOUT,
) -> ClaudeResult:
    """Resume an existing Claude session with a follow-up prompt."""
    cmd = [
        "claude",
        "-p", prompt,
        "--output-format", "stream-json",
        "--verbose",
        "--resume", session_id,
        "--model", model,
        "--permission-mode", "bypassPermissions",
    ]

    _log(f"Resuming session {session_id[:8]}...: {prompt[:80]}...")

    start = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
        cwd=cwd,
    )

    result_data: dict | None = None
    last_activity = time.monotonic()
    activity_count = 0
    buf = b""

    try:
        assert proc.stdout is not None
        fd = proc.stdout.fileno()

        while True:
            ready, _, _ = select.select([fd], [], [], 5.0)

            now = time.monotonic()
            elapsed = now - start
            idle = now - last_activity

            if elapsed >= timeout:
                proc.kill()
                proc.wait()
                return ClaudeResult(
                    success=False, output="", cost_usd=0,
                    duration_s=elapsed,
                    error=f"Timeout after {elapsed:.0f}s",
                    session_id=session_id,
                    had_activity=activity_count > 0,
                )

            if idle >= stall_timeout:
                proc.kill()
                proc.wait()
                return ClaudeResult(
                    success=False, output="", cost_usd=0,
                    duration_s=elapsed,
                    error=f"Stalled: no output for {idle:.0f}s",
                    session_id=session_id,
                    had_activity=activity_count > 0,
                )

            if not ready:
                continue

            chunk = proc.stdout.read1(8192) if hasattr(proc.stdout, 'read1') else proc.stdout.read(8192)
            if not chunk:
                break

            last_activity = time.monotonic()
            buf += chunk

            while b"\n" in buf:
                line_bytes, buf = buf.split(b"\n", 1)
                line = line_bytes.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                etype = event.get("type", "")
                if etype == "assistant":
                    _process_assistant_event(event)
                    activity_count += 1
                elif etype == "result":
                    result_data = event

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

        error_msg = None
        if is_error:
            error_msg = result_text or "Unknown error"

        return ClaudeResult(
            success=not is_error,
            output=result_text,
            cost_usd=cost,
            duration_s=duration,
            error=error_msg,
            session_id=session_id,
            had_activity=activity_count > 0,
        )

    stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
    return ClaudeResult(
        success=False, output="", cost_usd=0,
        duration_s=duration,
        error=stderr or f"No result event, exit code {proc.returncode}",
        session_id=session_id,
        had_activity=activity_count > 0,
    )
