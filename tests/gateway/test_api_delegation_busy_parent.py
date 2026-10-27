"""A parent API session that is mid-turn must not cost a finished helper its delivery.

On a stateless ``api_server`` surface a delegation result is persisted as a detached
delivery row, and the session store refuses that row while the parent turn holds its
turn lease (``SessionTurnLeaseLostError``) — by design. That refusal is transient: the
lease is released (or goes stale) when the turn ends. Counting it as a spent durable
delivery attempt meant any parent turn longer than the retry backoff (~4 minutes)
converged the finished result to ``dropped`` and the client never got its follow-up turn.
"""

import asyncio
import queue
import sqlite3
from collections import OrderedDict
from types import SimpleNamespace

import pytest

from gateway import run_notifications
from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.wake import WakeNotAccepted


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import tools.process_registry as pr_module

    monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", tmp_path / "processes.json")
    registry = pr_module.ProcessRegistry()
    monkeypatch.setattr(pr_module, "process_registry", registry)
    return registry


def _api_runner(db):
    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.API_SERVER: SimpleNamespace(
        supports_async_delivery=False, _ensure_session_db=lambda: db)}
    runner.session_store = SimpleNamespace(_ensure_loaded=lambda: None, _entries={})
    runner._session_source_cache = {}
    runner._completion_delivery_lock = __import__("threading").Lock()
    runner._completion_deliveries_inflight = set()
    runner._completion_deliveries_delivered = OrderedDict()
    runner._completion_delivery_retention = 2048
    runner._background_tasks = set()
    return runner


def _api_event(delegation_id="deleg_busy_parent"):
    return {
        "type": "async_delegation",
        "delegation_id": delegation_id,
        "session_key": "api-parent",
        "origin_session_id": "api-parent",
        "platform": "api_server",
        "goal": "Gather facts",
        "status": "completed",
        "summary": "HELPER_RESULT",
        "api_calls": 1,
        "duration_seconds": 5.0,
        "dispatched_at": 1000.0,
        "completed_at": 1005.0,
    }


def _ledger_row(tmp_path, delegation_id):
    conn = sqlite3.connect(tmp_path / "state.db")
    try:
        return conn.execute("SELECT delivery_state, delivery_attempts FROM async_delegations "
                            "WHERE delegation_id=?", (delegation_id,)).fetchone()
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_busy_parent_refusal_is_not_accepted_and_not_permanent(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "sessions.db")
    db.create_session("api-parent", source="api_server")
    assert db.acquire_session_turn_lease("api-parent", "client-turn", wait_seconds=0)
    runner = _api_runner(db)
    with pytest.raises(WakeNotAccepted) as exc:
        await runner._inject_watch_notification("[RESULT]", _api_event(), raise_not_accepted=True)
    assert exc.value.reason == "turn_lease"
    assert exc.value.permanent is False
    assert db.get_messages("api-parent") == []


@pytest.mark.asyncio
async def test_other_persist_failures_stay_plain_retryable(tmp_path, monkeypatch):
    import gateway.wake as wake_mod

    async def fail_persist(adapter, *, text, session_id, evt=None):
        raise RuntimeError("state.db unavailable")

    monkeypatch.setattr(wake_mod, "persist_delegation_delivery", fail_persist)
    runner = _api_runner(None)
    assert await runner._inject_watch_notification("[RESULT]", _api_event()) is False


def test_result_survives_a_parent_turn_longer_than_the_delivery_budget(
    monkeypatch, tmp_path, isolated_registry,
):
    from hermes_state import SessionDB
    from tools import async_delegation
    from tests.module_clock import install_module_clock

    db = SessionDB(db_path=tmp_path / "sessions.db")
    db.create_session("api-parent", source="api_server")
    assert db.acquire_session_turn_lease("api-parent", "client-turn", wait_seconds=0)

    evt = _api_event()
    async_delegation._persist_dispatch({
        "delegation_id": evt["delegation_id"], "session_key": evt["session_key"],
        "origin_ui_session_id": "", "parent_session_id": None, "dispatched_at": evt["dispatched_at"],
    })
    async_delegation._persist_completion(evt, {"status": "completed", "summary": evt["summary"]})

    q = queue.Queue()
    monkeypatch.setattr(isolated_registry, "completion_queue", q)
    q.put(evt)
    runner = _api_runner(db)

    clock = install_module_clock(monkeypatch, run_notifications)
    ticks = {"n": 0}
    release_after = async_delegation._MAX_DELIVERY_ATTEMPTS * 3  # well past the old budget

    async def _tick(_delay):
        ticks["n"] += 1
        clock.now += 60  # every backoff step is due on the next tick
        if ticks["n"] == release_after:
            # The parent turn finally ends; the still-pending result must land now.
            db.release_session_turn_lease("api-parent", "client-turn")
            state, attempts = _ledger_row(tmp_path, evt["delegation_id"])
            assert state == "pending", "a busy parent must never drop a finished result"
            assert attempts < async_delegation._MAX_DELIVERY_ATTEMPTS
        if ticks["n"] >= release_after + 3:
            runner._running = False

    monkeypatch.setattr(asyncio, "sleep", _tick)
    asyncio.run(runner._async_delegation_watcher(interval=0))

    assert _ledger_row(tmp_path, evt["delegation_id"])[0] == "delivered"
    rows = [m for m in db.get_messages("api-parent") if m.get("display_kind") == "async_delegation_complete"]
    assert len(rows) == 1
