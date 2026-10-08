"""A timed-out tool inside a desktop approval batch must NOT end the turn.

Regression for the "Operation interrupted." report: on the desktop, adjacent terminal
calls are prepared as one approval batch, and when one of them timed out the sequential
runner called ``agent.interrupt(...)`` — the whole turn ended with no reply at all, the
transcript got a bare "Operation interrupted." placeholder, and the user saw the session
stop for no visible reason (a single interactive command that waits on input was enough:
it hangs until the 420s sequential deadline).

Wanted behavior: abandon the REST of that batch (its unstarted calls are skipped, each
keeping a matching tool result) and let the model keep its turn, receive the timeout
result and decide — retry, change approach, or ask the user. That is already how the
non-prepared sequential path behaves (see test_sequential_tool_timeout.py).
"""

import concurrent.futures
import threading
from types import SimpleNamespace

import pytest

import agent.tool_executor as tool_executor
from agent.tool_executor import (
    _ManagedToolResult,
    _batch_abandon_reason,
    _run_sequential_tool_execution_middleware,
)


class _FakeAgent:
    def __init__(self):
        self._tool_worker_threads = set()
        self._tool_worker_threads_lock = threading.Lock()
        self._interrupt_requested = False
        self._tool_batch_abandoned = False
        self._tool_batch_abandoned_reason = ""
        self.activity = []

    def _touch_activity(self, msg):
        self.activity.append(msg)


class _Gate:
    def excluded_seconds(self):
        return 0.0


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setattr(tool_executor, "_SEQUENTIAL_INTERRUPT_POLL_SECONDS", 0.05)
    monkeypatch.setattr(tool_executor, "_emit_terminal_post_tool_call", lambda agent, **kw: None)


def _wedged_middleware(never: threading.Event):
    def _run(agent, **kwargs):
        never.wait(30)  # never returns within the test deadline
        return _ManagedToolResult(
            result="late result", args={}, middleware_trace=[], blocked=False, dispatched=True,
        )

    return _run


def _prepared_slot(closed: list):
    return SimpleNamespace(
        batch=SimpleNamespace(
            close=lambda: closed.append(True), authorization_gate=_Gate(), executor=None,
        ),
        tids=[],
        future=concurrent.futures.Future(),  # never completes
    )


def test_prepared_batch_timeout_abandons_batch_instead_of_ending_turn(monkeypatch):
    agent = _FakeAgent()
    closed: list = []
    never = threading.Event()

    monkeypatch.setattr("agent.terminal_approval_batch.take_prepared_call",
                        lambda call_id: _prepared_slot(closed))
    monkeypatch.setattr(tool_executor, "_resolve_sequential_tool_timeout", lambda: 0.3)
    monkeypatch.setattr(tool_executor, "_run_agent_tool_execution_middleware",
                        _wedged_middleware(never))

    try:
        managed = _run_sequential_tool_execution_middleware(
            agent,
            function_name="terminal",
            function_args={"command": "hermes tools"},
            effective_task_id="task",
            tool_call_id="call-1",
            execute=None,
        )
    finally:
        never.set()

    assert closed, "the prepared batch must still be closed (no overlapping commands)"
    assert agent._interrupt_requested is False, (
        "a tool timeout must not interrupt the turn — that is what produced a reply-less "
        "'Operation interrupted.' turn"
    )
    assert agent._tool_batch_abandoned is True
    assert "timed out" in agent._tool_batch_abandoned_reason
    assert "timed out" in str(managed.result)
    # The hint must tell the model HOW to recover, not just that it failed.
    assert "non-interactively" in str(managed.result)


def test_abandon_reason_is_none_without_an_abandoned_batch():
    assert _batch_abandon_reason(_FakeAgent()) is None

    agent = _FakeAgent()
    agent._tool_batch_abandoned = True
    assert _batch_abandon_reason(agent) == "previous tool did not complete"

    agent._tool_batch_abandoned_reason = "terminal timed out after 420.0s"
    assert _batch_abandon_reason(agent) == "terminal timed out after 420.0s"


def test_skipped_calls_of_an_abandoned_batch_name_the_timeout(monkeypatch):
    """The skipped result the model sees must blame the timeout, not a phantom user stop."""
    agent = _FakeAgent()
    agent._tool_batch_abandoned = True
    agent._tool_batch_abandoned_reason = "terminal timed out after 420.0s"
    agent.log_prefix = ""
    agent._vprint = lambda *a, **kw: None

    messages: list = []
    monkeypatch.setattr(tool_executor, "_flush_session_db_after_tool_progress",
                        lambda *a, **kw: True)
    monkeypatch.setattr(tool_executor, "make_tool_result_message",
                        lambda name, content, call_id, **kw: {
                            "role": "tool", "name": name, "content": content, "tool_call_id": call_id,
                        })

    call = SimpleNamespace(id="call-2", type="function",
                           function=SimpleNamespace(name="terminal", arguments="{}"))
    ok = tool_executor._skip_remaining_sequential(
        agent, messages, [call], "task",
        notice="remaining tool call(s)",
        content=(f"[Tool execution skipped — {{name}} was not started. "
                 f"{_batch_abandon_reason(agent)}.]"),
        banner="⚡ Tool batch abandoned:",
        flush_stage="skipped tool result",
    )

    assert ok is True
    assert len(messages) == 1
    assert "timed out after 420.0s" in messages[0]["content"]
    assert "user sent a new message" not in messages[0]["content"]


def test_desktop_batch_timeout_keeps_the_turn_with_real_command(tmp_path, monkeypatch):
    """End-to-end proof on the REAL desktop approval-batch path with a command that hangs.

    Mirrors the reported failure: two adjacent terminal calls on the desktop, the first one
    a command that never returns (interactive prompt / long-running child). Before the fix
    the runner interrupted the turn, so the transcript ended in tool results plus one
    reply-less "Operation interrupted."; now the batch is abandoned and the turn continues.
    """
    import json
    import queue
    from contextlib import ExitStack
    from unittest.mock import patch as _patch

    from gateway.session_context import clear_session_vars
    from run_agent import AIAgent
    from tools import approval
    from tools.terminal_scope import reset_terminal_scope, set_terminal_scope
    from tools.terminal_tool import TERMINAL_SCHEMA

    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    monkeypatch.setenv("HERMES_CONCURRENT_TOOL_TIMEOUT_S", "1.0")
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *a, **kw: None)

    with (
        _patch("model_tools.get_tool_definitions",
               return_value=[{"type": "function", "function": TERMINAL_SCHEMA}]),
        _patch("model_tools.check_toolset_requirements", return_value={}),
        _patch("agent.process_bootstrap.OpenAI"),
        _patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1",
                        quiet_mode=True, skip_context_files=True, skip_memory=True,
                        platform="desktop")

    def _terminal_call(call_id, command):
        return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(
            name="terminal", arguments=json.dumps({"command": command})))

    key = "desktop-batch-timeout"
    published: queue.Queue = queue.Queue()
    approval.register_gateway_notify(key, published.put)
    from tui_gateway import server
    monkeypatch.setattr(server, "_sessions", {key: {
        "session_key": key, "source": "desktop", "agent": agent, "cwd": str(tmp_path)}})
    tokens = server._set_session_context(key)
    agent._flush_messages_to_session_db = lambda rows, **kw: True

    messages: list = []
    try:
        with ExitStack() as scope:
            scope.callback(
                reset_terminal_scope,
                set_terminal_scope({"TERMINAL_ENV": "local", "TERMINAL_CWD": str(tmp_path)}),
            )
            agent._execute_tool_calls(
                SimpleNamespace(tool_calls=[
                    _terminal_call("hung", "sleep 8"),
                    _terminal_call("next", "echo SECOND"),
                ]),
                messages, key,
            )
    finally:
        # Same teardown the approval-batch suite uses: drop the notify hook, release the
        # real terminal env this test created (a leaked env poisons later tests), and
        # clear the bound session context.
        from tools.terminal_tool_lifecycle import cleanup_vm
        approval.unregister_gateway_notify(key)
        cleanup_vm(key)
        clear_session_vars(tokens)

    assert agent._interrupt_requested is False, (
        "a hanging command must not interrupt the turn — that is the reported "
        "reply-less 'Operation interrupted.' session stop"
    )
    assert [message.get("tool_call_id") for message in messages] == ["hung", "next"]
    assert "timed out" in messages[0]["content"]
    assert "timed out" in messages[1]["content"]
    assert "user sent a new message" not in messages[1]["content"]
