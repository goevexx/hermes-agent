"""Tests for the Tower routing boundary: classification, prompt form,
no-double-graph invariant, and fail-closed fallback.

Acceptance criteria covered:
AC-1  classify_for_tower is deterministic (skill marker, assignee, metadata).
AC-2  Tower spawn uses `claude -p` with literal `/tower` in the prompt.
AC-3  Tower-routed tasks bypass _default_spawn's normal leaf-graph path.
AC-4  classify_for_tower returns False for clear single-slice tasks.
AC-5  assert_codex_blocked() raises on any Codex call attempt.
AC-6  Fail-closed: RuntimeError when `claude` is not on PATH.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_workspace as kbw
from hermes_cli import kanban_tower_router as tower


# ---------------------------------------------------------------------------
# Minimal Task stub so tests don't need a live DB for pure-logic checks
# ---------------------------------------------------------------------------

def _make_task(
    *,
    id: str = "t_test",
    title: str = "Test task",
    body: Optional[str] = None,
    assignee: Optional[str] = "researcher",
    skills: Optional[list] = None,
) -> SimpleNamespace:
    """Build a task-like namespace sufficient for classification functions."""
    return SimpleNamespace(id=id, title=title, body=body, assignee=assignee, skills=skills)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    from pathlib import Path
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ===========================================================================
# AC-1: Classification — deterministic routing signals
# ===========================================================================

class TestClassifyForTower:
    def test_skill_marker_routes_to_tower(self):
        task = _make_task(skills=["tower-supervisor", "other-skill"])
        assert tower.classify_for_tower(task) is True

    def test_skill_marker_alone_routes_to_tower(self):
        task = _make_task(skills=["tower-supervisor"])
        assert tower.classify_for_tower(task) is True

    def test_assignee_tower_routes_to_tower(self):
        task = _make_task(assignee="tower")
        assert tower.classify_for_tower(task) is True

    def test_assignee_tower_case_insensitive(self):
        task = _make_task(assignee="Tower")
        assert tower.classify_for_tower(task) is True

    def test_json_body_route_tower_routes_to_tower(self):
        body = 'Some description.\n{"route": "tower"}\nMore text.'
        task = _make_task(body=body)
        assert tower.classify_for_tower(task) is True

    def test_json_body_embedded_route_tower(self):
        body = json.dumps({"route": "tower", "extra": "field"})
        task = _make_task(body=body)
        assert tower.classify_for_tower(task) is True

    # AC-4: Single clear slices must NOT be routed to Tower
    def test_no_signal_stays_on_kanban(self):
        task = _make_task(skills=["research"], assignee="researcher")
        assert tower.classify_for_tower(task) is False

    def test_normal_assignee_not_tower(self):
        task = _make_task(assignee="worker", skills=None)
        assert tower.classify_for_tower(task) is False

    def test_empty_skills_not_tower(self):
        task = _make_task(skills=[])
        assert tower.classify_for_tower(task) is False

    def test_none_skills_not_tower(self):
        task = _make_task(skills=None)
        assert tower.classify_for_tower(task) is False

    def test_body_without_tower_key_not_tower(self):
        body = json.dumps({"route": "kanban"})
        task = _make_task(body=body)
        assert tower.classify_for_tower(task) is False

    def test_no_body_not_tower(self):
        task = _make_task(body=None)
        assert tower.classify_for_tower(task) is False


# ===========================================================================
# AC-2: Prompt form — `claude -p` with literal `/tower` in prompt
# ===========================================================================

class TestTowerPromptForm:
    def test_prompt_contains_slash_tower_literally(self):
        task = _make_task(id="t_abc", title="Big initiative", body="Multi-repo refactor.")
        prompt = tower.build_tower_supervisor_prompt(task, "my-board", "/tmp/ws")
        assert "/tower" in prompt

    def test_prompt_starts_with_slash_tower(self):
        task = _make_task(id="t_abc", title="Big initiative", body="Multi-repo refactor.")
        prompt = tower.build_tower_supervisor_prompt(task, "my-board", "/tmp/ws")
        assert prompt.lstrip().startswith("/tower")

    def test_prompt_references_task_id(self):
        task = _make_task(id="t_xyz99", title="test", body="body text")
        prompt = tower.build_tower_supervisor_prompt(task, "board-slug", "/tmp/ws")
        assert "t_xyz99" in prompt

    def test_prompt_contains_no_fallback_provider_instruction(self):
        task = _make_task(id="t_p", title="T", body="B")
        prompt = tower.build_tower_supervisor_prompt(task, "board", "/tmp/ws")
        assert "Never select a fallback" not in prompt  # this is the normal worker wording
        assert "no provider fallback" not in prompt.lower()
        # The tower prompt MUST forbid fallback
        assert "never switch provider" in prompt.lower() or "no provider fallback" in prompt.lower() or "never switch" in prompt.lower()

    def test_prompt_contains_board_slug_in_lifecycle_commands(self):
        task = _make_task(id="t_lc", title="T", body="B")
        prompt = tower.build_tower_supervisor_prompt(task, "drs-board", "/tmp/ws")
        assert "'drs-board'" in prompt or '"drs-board"' in prompt

    def test_prompt_forbids_codex_and_openrouter(self):
        task = _make_task(id="t_pf", title="T", body="B")
        prompt = tower.build_tower_supervisor_prompt(task, "board", "/tmp/ws")
        assert "Codex" in prompt or "codex" in prompt.lower()
        assert "OpenRouter" in prompt or "openrouter" in prompt.lower()

    def test_prompt_contains_rate_limit_block_instruction(self):
        task = _make_task(id="t_rl", title="T", body="B")
        prompt = tower.build_tower_supervisor_prompt(task, "board", "/tmp/ws")
        # Must tell worker to block on rate limit, not switch provider
        assert "rate limit" in prompt.lower() or "rate-limit" in prompt.lower()
        assert "block" in prompt.lower()


# ===========================================================================
# AC-3: No-double-graph — Tower path bypasses normal leaf-graph spawn
# ===========================================================================

class TestNoDoubleGraph:
    def test_tower_task_dispatched_via_tower_not_normal_spawn(
        self, kanban_home, monkeypatch
    ):
        """classify_for_tower → spawn_tower_supervisor is called; _worker_argv is NOT."""
        tower_calls = []
        normal_calls = []

        def fake_spawn_tower(task, workspace, *, board=None):
            tower_calls.append(task.id)
            return 9999

        monkeypatch.setattr(
            "hermes_cli.kanban_tower_router.classify_for_tower",
            lambda t: t.skills and "tower-supervisor" in t.skills,
        )
        monkeypatch.setattr(
            "hermes_cli.kanban_tower_router.spawn_tower_supervisor",
            fake_spawn_tower,
        )
        monkeypatch.setattr(kbd, "_worker_argv", lambda *a, **kw: normal_calls.append("called"))

        with kbc.connect_closing() as conn:
            tid = kb.create_task(
                conn,
                title="Multi-repo migration",
                assignee="tower",
                skills=["tower-supervisor"],
            )
            task = kb.get_task(conn, tid)
        assert task is not None
        workspace = str(kbw.resolve_workspace(task))

        pid = kbd._default_spawn(task, workspace)

        assert pid == 9999, "Tower supervisor PID must be returned"
        assert tid in tower_calls, "spawn_tower_supervisor must be called"
        assert len(normal_calls) == 0, "_worker_argv must NOT be called for Tower tasks"

    def test_non_tower_task_uses_normal_backend(
        self, kanban_home, monkeypatch
    ):
        """A plain task still goes through the configured backend (not Tower)."""
        tower_calls = []

        def fake_spawn_tower(task, workspace, *, board=None):
            tower_calls.append(task.id)
            return 1234

        monkeypatch.setattr(
            "hermes_cli.kanban_tower_router.spawn_tower_supervisor",
            fake_spawn_tower,
        )
        # Pretend claude_cli backend with a fake Popen
        captured = {}

        class FakeProc:
            pid = 55555
            stdin = None

        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"kanban": {"worker_backend": "claude_cli", "claude_cli_model": "sonnet", "claude_cli_permission_mode": "auto"}},
        )
        monkeypatch.setattr(kbd, "_resolve_claude_cli_argv", lambda: ["fake-claude"])
        monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: FakeProc())

        with kbc.connect_closing() as conn:
            tid = kb.create_task(
                conn,
                title="Simple single task",
                assignee="researcher",
            )
            task = kb.get_task(conn, tid)
        assert task is not None
        workspace = str(kbw.resolve_workspace(task))

        pid = kbd._default_spawn(task, workspace)

        assert pid == 55555, "Normal worker PID must be returned"
        assert len(tower_calls) == 0, "Tower path must NOT be taken for plain tasks"


# ===========================================================================
# AC-5: Codex fail-closed
# ===========================================================================

class TestCodexGuard:
    def test_assert_codex_blocked_raises(self):
        with pytest.raises(RuntimeError, match="Codex is fail-closed"):
            tower.assert_codex_blocked()

    def test_assert_codex_blocked_message_mentions_jev_exception(self):
        try:
            tower.assert_codex_blocked()
        except RuntimeError as exc:
            assert "JEV" in str(exc), "Error must name the JEV exception so users know OpenRouter is not banned globally"

    def test_unknown_backend_refuses_fallback(self, monkeypatch):
        """_worker_backend_config blocks unrecognised backend names."""
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"kanban": {"worker_backend": "codex"}},
        )
        with pytest.raises(RuntimeError, match="only claude_cli is permitted"):
            kbd._worker_backend_config()

    def test_openai_codex_backend_refused(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"kanban": {"worker_backend": "openai-codex"}},
        )
        with pytest.raises(RuntimeError, match="only claude_cli is permitted"):
            kbd._worker_backend_config()


# ===========================================================================
# AC-6: Fail-closed — `claude` not on PATH
# ===========================================================================

class TestFailClosed:
    def test_spawn_tower_supervisor_raises_when_claude_not_on_path(
        self, kanban_home, monkeypatch
    ):
        """spawn_tower_supervisor must raise RuntimeError; must not silently fall back."""

        def no_claude():
            raise RuntimeError(
                "kanban.worker_backend is claude_cli but `claude` is not on PATH; "
                "refusing to fall back to Hermes/Codex"
            )

        with kbc.connect_closing() as conn:
            tid = kb.create_task(
                conn,
                title="Tower task missing claude",
                assignee="tower",
                skills=["tower-supervisor"],
            )
            task = kb.get_task(conn, tid)
        assert task is not None
        workspace = str(kbw.resolve_workspace(task))

        with pytest.raises(RuntimeError, match="not on PATH|refusing to fall back|failed to start"):
            tower.spawn_tower_supervisor(
                task,
                workspace,
                _resolve_claude_cli_argv=no_claude,
            )

    def test_spawn_tower_supervisor_raises_on_popen_failure(
        self, kanban_home, monkeypatch
    ):
        """FileNotFoundError from Popen is wrapped as RuntimeError (no silent fallback)."""
        import subprocess as sp

        def fake_popen(*a, **kw):
            raise FileNotFoundError("claude not found")

        monkeypatch.setattr(sp, "Popen", fake_popen)

        with kbc.connect_closing() as conn:
            tid = kb.create_task(
                conn,
                title="Tower task popen failure",
                assignee="tower",
                skills=["tower-supervisor"],
            )
            task = kb.get_task(conn, tid)
        assert task is not None
        workspace = str(kbw.resolve_workspace(task))

        with pytest.raises(RuntimeError, match="failed to start Tower supervisor"):
            tower.spawn_tower_supervisor(
                task,
                workspace,
                _resolve_claude_cli_argv=lambda: ["fake-claude"],
            )

    def test_rate_limit_error_produces_block_not_fallback(self):
        """Prompt explicitly instructs worker to block on rate limit, never switch provider."""
        task = _make_task(id="t_rl2", title="Tower task", body="Important work")
        prompt = tower.build_tower_supervisor_prompt(task, "test-board", "/tmp/ws")
        # Must instruct block with capability kind
        assert "block" in prompt.lower()
        assert "capability" in prompt.lower()
        # The prompt mentions --fallback-model only to prohibit it
        # (e.g. "Do NOT invoke ... Claude's --fallback-model")
        assert "do not" in prompt.lower() or "never" in prompt.lower()

    def test_spawn_tower_uses_literal_minus_p_flag(
        self, kanban_home, monkeypatch
    ):
        """Tower supervisor must use `claude -p`, not `hermes -p` or `--print`."""
        captured = {}

        class FakeProc:
            pid = 77777

            class stdin:
                @staticmethod
                def write(data):
                    captured["prompt"] = data

                @staticmethod
                def close():
                    captured["closed"] = True

        import subprocess as sp
        monkeypatch.setattr(sp, "Popen", lambda cmd, **kw: (captured.__setitem__("cmd", cmd) or FakeProc()))

        with kbc.connect_closing() as conn:
            tid = kb.create_task(
                conn,
                title="Tower literal -p",
                assignee="tower",
                skills=["tower-supervisor"],
            )
            task = kb.get_task(conn, tid)
        assert task is not None
        workspace = str(kbw.resolve_workspace(task))

        tower.spawn_tower_supervisor(
            task,
            workspace,
            _resolve_claude_cli_argv=lambda: ["fake-claude"],
        )

        cmd = captured.get("cmd", [])
        assert "fake-claude" in cmd, "claude binary must be in argv"
        assert "-p" in cmd, "literal -p flag required"
        assert "--print" not in cmd, "legacy --print flag must not be used"
        prompt = captured.get("prompt", b"").decode("utf-8")
        assert "/tower" in prompt, "Literal /tower must appear in the prompt sent to Claude Code"


# ===========================================================================
# Regression tests: all-path Codex guard (t_a8262bd1)
# ===========================================================================

class TestCodexGuardRegressions:
    """Explicit regression coverage for the defects found in t_a8262bd1.

    Each test sets one of the forbidden configurations and asserts that no
    Hermes/Codex worker argv or process is produced.
    """

    # -----------------------------------------------------------------------
    # _worker_backend_config() must reject the legacy "hermes" value
    # -----------------------------------------------------------------------

    def test_hermes_backend_rejected(self, monkeypatch):
        """worker_backend=hermes must now raise, not spawn a Hermes worker."""
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"kanban": {"worker_backend": "hermes"}},
        )
        with pytest.raises(RuntimeError, match="only claude_cli is permitted"):
            kbd._worker_backend_config()

    def test_hermes_backend_default_is_now_claude_cli(self, monkeypatch):
        """Omitting worker_backend defaults to claude_cli, not hermes."""
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"kanban": {}},
        )
        backend, _, _ = kbd._worker_backend_config()
        assert backend == "claude_cli"

    # -----------------------------------------------------------------------
    # provider_override=openai-codex must block before Popen
    # -----------------------------------------------------------------------

    def test_provider_override_openai_codex_blocked(self, kanban_home, monkeypatch):
        """provider_override=openai-codex must not produce a worker process."""
        spawn_calls: list[str] = []

        class FakeProc:
            pid = 99991
            stdin = None

        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"kanban": {"worker_backend": "claude_cli"}},
        )
        monkeypatch.setattr(kbd, "_resolve_claude_cli_argv", lambda: ["fake-claude"])
        import subprocess as sp
        monkeypatch.setattr(
            sp, "Popen",
            lambda *a, **kw: (spawn_calls.append("spawned") or FakeProc()),
        )

        with kbc.connect_closing() as conn:
            tid = kb.create_task(
                conn,
                title="Task with Codex provider override",
                assignee="researcher",
                provider_override="openai-codex",
                model_override="codex-latest",
            )
            task = kb.get_task(conn, tid)
        assert task is not None
        workspace = str(kbw.resolve_workspace(task))

        with pytest.raises(RuntimeError, match="Codex is fail-closed"):
            kbd._default_spawn(task, workspace)

        assert len(spawn_calls) == 0, "No process must be spawned with Codex provider override"

    # -----------------------------------------------------------------------
    # model_override=gpt-5.6-sol must block before Popen
    # -----------------------------------------------------------------------

    def test_model_override_gpt_blocked(self, kanban_home, monkeypatch):
        """model_override=gpt-5.6-sol must not produce a worker process."""
        spawn_calls: list[str] = []

        class FakeProc:
            pid = 99992
            stdin = None

        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"kanban": {"worker_backend": "claude_cli"}},
        )
        monkeypatch.setattr(kbd, "_resolve_claude_cli_argv", lambda: ["fake-claude"])
        import subprocess as sp
        monkeypatch.setattr(
            sp, "Popen",
            lambda *a, **kw: (spawn_calls.append("spawned") or FakeProc()),
        )

        with kbc.connect_closing() as conn:
            tid = kb.create_task(
                conn,
                title="Task with GPT model override",
                assignee="researcher",
                model_override="gpt-5.6-sol",
            )
            task = kb.get_task(conn, tid)
        assert task is not None
        workspace = str(kbw.resolve_workspace(task))

        with pytest.raises(RuntimeError, match="Codex is fail-closed"):
            kbd._default_spawn(task, workspace)

        assert len(spawn_calls) == 0, "No process must be spawned with GPT model override"

    # -----------------------------------------------------------------------
    # provider_override=openrouter must block (OpenRouter is JEV-only)
    # -----------------------------------------------------------------------

    def test_openrouter_provider_blocked_in_kanban(self, kanban_home, monkeypatch):
        """provider_override=openrouter must not produce a worker process."""
        spawn_calls: list[str] = []

        class FakeProc:
            pid = 99993
            stdin = None

        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"kanban": {"worker_backend": "claude_cli"}},
        )
        monkeypatch.setattr(kbd, "_resolve_claude_cli_argv", lambda: ["fake-claude"])
        import subprocess as sp
        monkeypatch.setattr(
            sp, "Popen",
            lambda *a, **kw: (spawn_calls.append("spawned") or FakeProc()),
        )

        with kbc.connect_closing() as conn:
            tid = kb.create_task(
                conn,
                title="Task with OpenRouter provider",
                assignee="researcher",
                provider_override="openrouter",
                model_override="anthropic/claude-sonnet-4-5",
            )
            task = kb.get_task(conn, tid)
        assert task is not None
        workspace = str(kbw.resolve_workspace(task))

        with pytest.raises(RuntimeError, match="Codex is fail-closed"):
            kbd._default_spawn(task, workspace)

        assert len(spawn_calls) == 0, "OpenRouter must not spawn a Kanban worker"

    # -----------------------------------------------------------------------
    # Verify _worker_argv is unreachable: hermes backend is fully dead
    # -----------------------------------------------------------------------

    def test_no_hermes_worker_argv_producible(self, monkeypatch):
        """_worker_argv must never be called by _worker_backend_config + dispatch.

        Setting worker_backend=hermes must raise before _worker_argv is
        reached, producing no hermes worker command.
        """
        hermes_argv_calls: list[str] = []
        original_worker_argv = kbd._worker_argv

        def spy_worker_argv(*args, **kwargs):
            hermes_argv_calls.append("called")
            return original_worker_argv(*args, **kwargs)

        monkeypatch.setattr(kbd, "_worker_argv", spy_worker_argv)
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"kanban": {"worker_backend": "hermes"}},
        )

        with pytest.raises(RuntimeError, match="only claude_cli is permitted"):
            kbd._worker_backend_config()

        assert len(hermes_argv_calls) == 0, "_worker_argv must never be reached via hermes backend"
