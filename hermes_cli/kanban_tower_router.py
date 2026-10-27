"""Tower routing boundary: classify Kanban tasks for Tower vs. normal Kanban path.

Deterministic rules — no LLM call at routing time. Tower handles multi-step,
uncertain, long-term or cross-repo work; normal Kanban handles single clear
slices executed directly via the ``claude -p`` backend.

## Routing signals (any one is sufficient to route to Tower)

1. ``task.skills`` contains ``"tower-supervisor"``  ← explicit operator flag
2. ``task.assignee == "tower"``                     ← special assignee lane
3. ``task`` metadata JSON contains ``{"route": "tower"}``

## No-double-graph invariant

A Tower-routed task must NEVER fan out a parallel leaf execution graph inside
Kanban. The Tower supervisor task is the single unit of work that Hermes tracks;
all inner Archon/ogsp-* nodes are Tower-internal. Callers of
``spawn_tower_supervisor`` MUST skip the normal ``_default_spawn`` path.

## Fail-closed

If ``claude`` is not on PATH when routing to Tower, a ``RuntimeError`` is raised.
No fallback to Hermes provider or Codex is attempted. Callers must catch this and
block the task rather than silently substituting another runner.

## Codex guard

Any Kanban code path that would invoke OpenAI Codex as a worker engine MUST call
``assert_codex_blocked()`` first. This function raises ``RuntimeError`` with a
clear message; it is intentionally not a soft warning.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from hermes_cli.kanban_db import Task

# Skills value that triggers Tower routing when found in task.skills.
TOWER_SKILL_MARKER = "tower-supervisor"

# Assignee name that routes to Tower.
TOWER_ASSIGNEE = "tower"

# Metadata key/value that explicitly flags Tower routing.
_TOWER_META_KEY = "route"
_TOWER_META_VALUE = "tower"


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_for_tower(task: "Task") -> bool:
    """Return True iff this task must route through Tower.

    Routing signals (any one is sufficient):

    * ``task.skills`` contains ``TOWER_SKILL_MARKER`` (``"tower-supervisor"``)
    * ``task.assignee == "tower"``
    * ``task`` metadata JSON (from the body's first JSON block or task.result)
      contains ``{"route": "tower"}``

    This function is intentionally deterministic and has no side-effects.
    """
    # Signal 1: explicit skill marker
    if task.skills and TOWER_SKILL_MARKER in task.skills:
        return True

    # Signal 2: special assignee lane
    if (task.assignee or "").strip().lower() == TOWER_ASSIGNEE:
        return True

    # Signal 3: metadata in task body
    if _body_requests_tower(task.body):
        return True

    return False


def _body_requests_tower(body: Optional[str]) -> bool:
    """Scan the task body for a ``route: tower`` JSON block."""
    if not body:
        return False
    import json, re

    # Accept a JSON object anywhere in the body that sets route=tower.
    for m in re.finditer(r"\{[^{}]*\}", body, re.DOTALL):
        try:
            obj = json.loads(m.group())
        except json.JSONDecodeError:
            continue
        if obj.get(_TOWER_META_KEY) == _TOWER_META_VALUE:
            return True
    return False


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_tower_supervisor_prompt(task: "Task", board_slug: str, workspace: str) -> str:  # noqa: ARG001 (workspace reserved for future cwd embedding)
    """Return the literal prompt body for ``claude -p "/tower ..."`` invocations.

    The prompt instructs Claude Code to invoke the Tower skill with the task's
    description as the flight plan goal, and to reconcile the outcome back into
    Kanban via the standard lifecycle commands.

    The ``/tower`` prefix is present literally — required because the Tower skill
    sets ``disable-model-invocation: true`` and will not activate without it.
    """
    task_ref = _tower_issue_ref(task)
    goal_text = (task.body or task.title or "").strip()

    return f"""\
/tower {task_ref}

You are running as the Hermes Kanban Tower supervisor for task {task.id}.

Tower goal:
{goal_text}

Work only on this task. Do not start competing Archon or ogsp-* runs outside
the Tower flight plan. Tower is the single source of truth for the inner graph;
Hermes tracks only the supervisor-level outcome.

Do NOT invoke another LLM provider, Hermes chat, Codex, OpenRouter, or
Claude's --fallback-model. Use your current first-party Claude Code session only.

If you hit a Claude Code rate limit, block the task — never switch provider.

When Tower finishes (all flight-plan nodes reach ``fertig`` or ``gescheitert``),
perform exactly one terminal transition against board {board_slug!r}:

* Tower plan completed:
  hermes kanban --board {board_slug!r} complete {task.id} \\
    --summary '<Tower outcome in 1-3 sentences>' \\
    --metadata '{{"tower_issue": "{task_ref}", "status": "completed"}}'

* Tower plan blocked on human input or unresolvable dependency:
  hermes kanban --board {board_slug!r} block \\
    --kind needs_input {task.id} \\
    '<specific blocker — Tower issue #, blocked node, reason>'

* Claude Code rate-limited / unavailable:
  hermes kanban --board {board_slug!r} block \\
    --kind capability {task.id} \\
    'Claude Code rate limit: fallback requires explicit user approval'

Do not report success unless the Tower flight plan was actually driven to
completion and every node has a terminal status.
"""


def _tower_issue_ref(task: "Task") -> str:
    """Extract a Tower GitLab issue number or return the task title as goal."""
    import re

    # If the body or metadata references a GitLab issue number, prefer it.
    for text in (task.body or "", task.title or ""):
        m = re.search(r"#(\d+)", text)
        if m:
            return f"#{m.group(1)}"
    return task.title or task.id


# ---------------------------------------------------------------------------
# Spawn
# ---------------------------------------------------------------------------

def spawn_tower_supervisor(
    task: "Task",
    workspace: str,
    *,
    board: Optional[str] = None,
    _resolve_claude_cli_argv=None,  # injectable for tests
) -> int:
    """Spawn ``claude -p "/tower ..."`` as a first-party Claude Code process.

    Returns the child PID. Fails closed: raises ``RuntimeError`` if
    ``claude`` is not on PATH or if spawning fails. Never falls back to
    Hermes/Codex/OpenRouter.

    The Tower supervisor task gets its full prompt on stdin so that task
    content never leaks into shell command-string interpolation.
    """
    import subprocess

    from hermes_cli import kanban_db as _kb
    from hermes_cli import kanban_db_dispatch as _kbd

    # Resolve the board slug for lifecycle commands embedded in the prompt.
    board_slug = _kb._normalize_board_slug(board) or _kb.get_current_board()

    # Build the prompt before touching the process layer so an encoding error
    # surfaces cleanly without a zombie child.
    prompt_bytes = build_tower_supervisor_prompt(task, board_slug, workspace).encode("utf-8")

    # Resolve the ``claude`` CLI — fail closed, never substitute.
    resolve = _resolve_claude_cli_argv or _kbd._resolve_claude_cli_argv
    cmd = [
        *resolve(),
        "-p",
        "--input-format", "text",
        "--output-format", "text",
        "--permission-mode", "auto",
    ]

    import os
    env = dict(os.environ)
    env.pop("HERMES_TUI", None)

    from tools.process_registry import systemd_user_bus_env
    env = systemd_user_bus_env(env)

    # Open the per-task log the same way the normal dispatcher does so the
    # log rotation / backup convention stays consistent.
    log_f = _kbd._open_worker_log(task, board)
    log_f.write(
        f"[hermes] launching Tower supervisor via Claude Code CLI for task {task.id}\n"
        f"[hermes] cmd: {cmd}\n".encode("utf-8")
    )
    log_f.flush()

    try:
        proc = subprocess.Popen(  # noqa: S603 — argv is a fixed list, no shell expansion
            cmd,
            cwd=workspace if os.path.isdir(workspace) else None,
            stdin=subprocess.PIPE,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
            creationflags=subprocess.CREATE_NO_WINDOW if _kb._IS_WINDOWS else 0,
        )
        if proc.stdin is not None:
            proc.stdin.write(prompt_bytes)
            proc.stdin.close()
    except (FileNotFoundError, BrokenPipeError) as exc:
        log_f.close()
        raise RuntimeError(
            f"failed to start Tower supervisor for task {task.id}; "
            "no provider fallback was attempted"
        ) from exc

    return proc.pid


# ---------------------------------------------------------------------------
# Codex guard
# ---------------------------------------------------------------------------

def assert_codex_blocked() -> None:
    """Raise RuntimeError if called — Codex is fail-closed in all Kanban paths.

    Call this at the top of any code path that would otherwise invoke OpenAI
    Codex as a worker engine. A clear error is better than a silent provider
    swap that violates the first-party Claude Code requirement.

    Rationale: the architecture mandates that all Kanban worker, judge, goal-
    mode, review, follow-up, sub-process and Tower paths run exclusively via
    the local ``claude -p`` first-party auth. OpenRouter is reserved for the
    separate JEV context only.
    """
    raise RuntimeError(
        "Codex is fail-closed in all Kanban worker paths. "
        "OpenAI/Codex must not be used for Kanban implementation, analysis, "
        "planning, research, review or Tower work. "
        "OpenRouter is permitted only for the separate JEV context. "
        "To unblock: obtain explicit user approval and activate the alternative "
        "provider only for that specific, scoped invocation."
    )
