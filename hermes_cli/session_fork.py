"""Fork an occupied session into an independent branch, at resume time.

``/branch`` forks the session the *current* process already holds. ``hermes -c`` /
``hermes --resume`` cannot: the target session's lease belongs to another live surface
(another window, the desktop app, a gateway turn), and per-session exclusivity is enforced
unconditionally, so the CLI prints "This chat is open in another Hermes window/terminal"
and exits.

This module turns that dead end into a usable branch: copy the transcript into a fresh
session owned by the caller and continue there. The original keeps running untouched — the
two conversations diverge from the fork point, which is exactly what a branch means.

Local patch (see skill ``hermes-session-takeover``).
"""

from __future__ import annotations

import logging
import os
from contextlib import suppress
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Message fields /branch copies verbatim onto the new row set (plus role / tool_name / api_content).
_COPY_KEYS = (
    "content",
    "tool_calls",
    "tool_call_id",
    "reasoning",
    "reasoning_content",
    "reasoning_details",
    "codex_reasoning_items",
    "codex_message_items",
    "timestamp",
)


def session_owner_pid(session_id: str) -> Optional[int]:
    """pid of ANOTHER live process holding *session_id*'s lease; None when free/unreadable.

    A pid equal to our own is not an owner for this purpose: re-claiming our own session is
    the caller's normal path, not a conflict to fork around.
    """
    target = str(session_id or "")
    if not target:
        return None
    try:
        from hermes_cli.active_sessions import active_session_registry_snapshot

        entries = active_session_registry_snapshot(strict=False)
    except Exception:  # unreadable registry: let the normal lease check decide
        logger.debug("active-session registry unavailable while probing %s", target, exc_info=True)
        return None
    for entry in entries:
        if str(entry.get("session_id") or "") != target:
            continue
        try:
            pid = int(entry.get("pid") or 0)
        except (TypeError, ValueError):
            continue
        if pid and pid != os.getpid():
            return pid
    return None


def fork_session_copy(
    session_id: str, *, source: Optional[str] = None, title_suffix: str = "分支"
) -> str:
    """Copy *session_id*'s transcript into a NEW session and return the new id.

    Mirrors ``/branch``'s seed (same copied fields) minus three things it must not do here:
    ending the parent row (that session is still live in another process), inheriting a live
    ``/yolo`` toggle (the branch is a different process's trust state), and reusing the
    parent's ``message_uid`` values — fresh rows get fresh uids from the store's triggers, so
    the branch never shares a durable message identity with the live parent.
    """
    from hermes_state import SessionDB
    from hermes_state_ids import new_session_id as mint_session_id

    db = SessionDB()
    try:
        parent: dict[str, Any] = db.get_session(session_id) or {}
        rows = db.get_messages(session_id)
        if not rows:
            raise ValueError(f"session {session_id} has no messages to fork")

        new_id = mint_session_id()
        title = db.get_session_title(session_id) or "branch"
        with suppress(Exception):
            title = db.get_next_title_in_lineage(title)
        if not title:
            title = f"branch {title_suffix}"

        parent_prompt = parent.get("system_prompt") if isinstance(parent, dict) else None
        db.create_session(
            session_id=new_id,
            source=source or os.environ.get("HERMES_SESSION_SOURCE", "cli"),
            model=parent.get("model") if isinstance(parent, dict) else None,
            parent_session_id=session_id,
            system_prompt=parent_prompt or None,
            cwd=parent.get("cwd") if isinstance(parent, dict) else None,
            # A stable provenance marker keeps the branch recognisable in /resume + /sessions
            # (and out of compression-continuation lookups, exactly like /branch).
            model_config={"_branched_from": session_id},
        )

        batch: list[dict[str, Any]] = []
        for row in rows:
            msg: dict[str, Any] = {
                "role": row.get("role") or "user",
                "tool_name": row.get("tool_name") or row.get("name"),
                "api_content": row.get("api_content"),
            }
            msg.update({key: row.get(key) for key in _COPY_KEYS})
            batch.append(msg)
        # Chunked: FTS triggers run per row, and a long transcript is a big transaction.
        db.append_messages_batch(new_id, batch, chunk_rows=500)
        with suppress(Exception):
            db.set_session_title(new_id, title)
        logger.info("Forked occupied session %s -> %s (%d messages)", session_id, new_id, len(batch))
        return new_id
    finally:
        with suppress(Exception):
            db.close()
