"""Tests for inline action button and extended element interactions."""

from __future__ import annotations

import asyncio
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kiro_crew.slack.interactions import (
    _approval_tool_label,
    _extract_selected_value,
    _mark_button_clicked,
)

# ---------------------------------------------------------------------------
# _mark_button_clicked
# ---------------------------------------------------------------------------


class TestMarkButtonClicked:
    def test_replaces_clicked_button_with_check(self) -> None:
        blocks = [
            {"type": "section", "text": {"type": "mrkdwn", "text": "Pick one"}},
            {
                "type": "actions",
                "elements": [
                    {"action_id": "btn_1", "text": {"type": "plain_text", "text": "A"}},
                    {"action_id": "btn_2", "text": {"type": "plain_text", "text": "B"}},
                ],
            },
        ]
        result = _mark_button_clicked(blocks, "btn_1", "A")
        assert result[0] == blocks[0]  # section unchanged
        assert result[1]["type"] == "context"
        assert "✓ A" in result[1]["elements"][0]["text"]
        assert result[2]["type"] == "actions"
        assert len(result[2]["elements"]) == 1
        assert result[2]["elements"][0]["action_id"] == "btn_2"

    def test_button_not_found_returns_unchanged(self) -> None:
        blocks = [
            {
                "type": "actions",
                "elements": [{"action_id": "btn_1"}],
            },
        ]
        result = _mark_button_clicked(blocks, "nonexistent", "X")
        assert result == blocks

    def test_last_button_removed_drops_empty_actions(self) -> None:
        blocks = [
            {
                "type": "actions",
                "elements": [{"action_id": "btn_1"}],
            },
        ]
        result = _mark_button_clicked(blocks, "btn_1", "Done")
        assert len(result) == 1
        assert result[0]["type"] == "context"


# ---------------------------------------------------------------------------
# _extract_selected_value
# ---------------------------------------------------------------------------


class TestExtractSelectedValue:
    def test_selected_option(self) -> None:
        action = {
            "selected_option": {
                "value": "high",
                "text": {"type": "plain_text", "text": "High"},
            }
        }
        assert _extract_selected_value(action) == ("high", "High")

    def test_selected_date(self) -> None:
        action = {"selected_date": "2026-04-08"}
        assert _extract_selected_value(action) == ("2026-04-08", "2026-04-08")

    def test_selected_time(self) -> None:
        action = {"selected_time": "14:30"}
        assert _extract_selected_value(action) == ("14:30", "14:30")

    def test_selected_date_time(self) -> None:
        action = {"selected_date_time": 1712592000}
        assert _extract_selected_value(action) == ("1712592000", "1712592000")

    def test_no_selection_returns_empty(self) -> None:
        assert _extract_selected_value({}) == ("", "")

    def test_malformed_selected_option_no_crash(self) -> None:
        """Defensive .get() prevents KeyError on malformed payloads."""
        action = {"selected_option": {"unexpected": True}}
        raw, display = _extract_selected_value(action)
        assert raw == ""
        assert display == ""


# ---------------------------------------------------------------------------
# _approval_tool_label
# ---------------------------------------------------------------------------


class TestApprovalToolLabel:
    def test_approval_tool_label_plain_footer(self) -> None:
        blocks = [{"type": "context", "elements": [{"text": ":lock: *Read /etc/passwd*"}]}]
        payload = {"message": {"blocks": blocks}}
        assert _approval_tool_label(payload) == "Read /etc/passwd"

    def test_approval_tool_label_with_source_tag(self) -> None:
        blocks = [
            {
                "type": "context",
                "elements": [{"text": ":lock: [subagent] *Read /etc/passwd*"}],
            }
        ]
        payload = {"message": {"blocks": blocks}}
        result = _approval_tool_label(payload)
        assert result == "Read /etc/passwd"
        assert "[subagent]" not in result
        assert "*" not in result

    def test_approval_tool_label_with_purpose(self) -> None:
        blocks = [
            {
                "type": "context",
                "elements": [{"text": ":lock: *Read /etc/passwd* — to check permissions"}],
            }
        ]
        payload = {"message": {"blocks": blocks}}
        result = _approval_tool_label(payload)
        assert result == "Read /etc/passwd"
        assert "to check permissions" not in result

    def test_approval_tool_label_with_tag_and_purpose(self) -> None:
        blocks = [
            {
                "type": "context",
                "elements": [{"text": ":lock: [subagent] *Read /etc/passwd* — to check"}],
            }
        ]
        payload = {"message": {"blocks": blocks}}
        result = _approval_tool_label(payload)
        assert result == "Read /etc/passwd"
        assert "[subagent]" not in result
        assert "*" not in result
        assert "to check" not in result

    def test_approval_tool_label_no_context_block(self) -> None:
        blocks = [
            {"type": "section", "text": {"type": "mrkdwn", "text": "Approve?"}},
            {"type": "actions", "elements": []},
        ]
        payload = {"message": {"blocks": blocks}}
        assert _approval_tool_label(payload) == ""

    def test_approval_tool_label_empty_elements(self) -> None:
        blocks = [{"type": "context", "elements": []}]
        payload = {"message": {"blocks": blocks}}
        assert _approval_tool_label(payload) == ""

    def test_approval_tool_label_blocks_not_a_list(self) -> None:
        assert _approval_tool_label({"message": {"blocks": "not-a-list"}}) == ""
        assert _approval_tool_label({"message": {}}) == ""
        assert _approval_tool_label({}) == ""

    def test_approval_tool_label_truncates_after_redaction(self) -> None:
        long_title = "A" * 250
        blocks = [{"type": "context", "elements": [{"text": f":lock: *{long_title}*"}]}]
        payload = {"message": {"blocks": blocks}}
        result = _approval_tool_label(payload)
        assert len(result) == 200
        assert result == long_title[:200]

        # Redaction must run BEFORE truncation, not after — otherwise a
        # credential/URL straddling the 200-char cut could survive half-redacted.
        credential = "sk-ant-" + "x" * 30
        credential_title = "y" * 190 + credential
        blocks_cred = [{"type": "context", "elements": [{"text": f":lock: *{credential_title}*"}]}]
        payload_cred = {"message": {"blocks": blocks_cred}}
        result_cred = _approval_tool_label(payload_cred)
        assert len(result_cred) == 200
        assert credential not in result_cred
        # A truncate-then-redact swap would leave the sliced fragment
        # "sk-ant-xxx" untouched (too short to match the credential pattern),
        # passing both asserts above for the wrong reason. Pin the exact
        # redact-then-truncate value to rule that swap out.
        assert result_cred == ("y" * 190 + "[REDACTED: credential]")[:200]

    def test_approval_tool_label_null_message(self) -> None:
        """``{"message": None}`` — key present but null, distinct from absent."""
        assert _approval_tool_label({"message": None}) == ""


# ---------------------------------------------------------------------------
# _handle_options — action button path
# ---------------------------------------------------------------------------


def _make_orch(post_ts: str = "9999.000") -> MagicMock:
    """Build a minimal mock orchestrator for _handle_options tests."""
    orch = MagicMock()
    orch.slack = AsyncMock()
    orch.slack.update_message = AsyncMock()
    orch.slack.post_message = AsyncMock(return_value=post_ts)
    orch.slack.delete_message = AsyncMock()
    orch.sessions = MagicMock()
    orch.ctx_builder = MagicMock()
    orch.cron_svc = MagicMock()
    orch.conv_log = MagicMock()
    orch.consolidator = MagicMock()
    orch.subagent_mgr = MagicMock()
    orch.task_runner = MagicMock()
    orch._handler_tasks = set()
    return orch


def _base_payload(
    value: str = "",
    action_id: str = "opt_0",
    thread_ts: str = "100.0",
    msg_ts: str = "200.0",
    blocks: list[dict] | None = None,
) -> tuple[dict, dict, str, str]:
    action = {"value": value, "action_id": action_id, "text": {"text": "Label"}}
    payload = {
        "user": {"id": "U123"},
        "team": {"id": "T1"},
        "message": {"ts": msg_ts, "thread_ts": thread_ts, "blocks": blocks or []},
    }
    return payload, action, "C1", msg_ts


@pytest.fixture
def orch_fixture(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Shared orchestrator mock with guaranteed cleanup via monkeypatch."""
    from kiro_crew.slack import interactions

    orch = _make_orch()
    monkeypatch.setattr(interactions, "_orch", orch)
    return orch


@pytest.mark.asyncio
async def test_action_button_happy_path(orch_fixture: MagicMock) -> None:
    """Button with action:: prefix routes payload to session as context."""
    from kiro_crew.slack import interactions

    orch = orch_fixture

    payload_json = '{"task":"deploy"}'
    payload, action, channel, msg_ts = _base_payload(
        value=f"action::{payload_json}", action_id="btn_deploy"
    )

    with patch.object(interactions, "handle_message", new_callable=AsyncMock) as mock_hm:
        await interactions._handle_options(payload, action, channel, msg_ts)
        # Let the created task run
        await asyncio.sleep(0)
        for t in list(orch._handler_tasks):
            await t

        mock_hm.assert_called_once()
        call_kwargs = mock_hm.call_args
        assert call_kwargs.kwargs["action_context"] is not None
        assert "Action button clicked" in call_kwargs.kwargs["action_context"]
        assert payload_json in call_kwargs.kwargs["action_context"]


@pytest.mark.asyncio
async def test_extended_element_happy_path(orch_fixture: MagicMock) -> None:
    """Extended element with action:: in action_id merges selected_value."""
    from kiro_crew.slack import interactions

    orch = orch_fixture

    base = json.dumps({"snooze": True})
    payload, action, channel, msg_ts = _base_payload(value="", action_id=f"action::{base}")
    action["selected_date"] = "2026-04-10"
    action["placeholder"] = {"text": "Pick date"}

    with patch.object(interactions, "handle_message", new_callable=AsyncMock) as mock_hm:
        await interactions._handle_options(payload, action, channel, msg_ts)
        await asyncio.sleep(0)
        for t in list(orch._handler_tasks):
            await t

        mock_hm.assert_called_once()
        ctx = mock_hm.call_args.kwargs["action_context"]
        assert "Action element selected" in ctx
        merged = json.loads(ctx.split(": ", 1)[1].split("]\n")[0])
        assert merged["selected_value"] == "2026-04-10"
        assert merged["snooze"] is True


@pytest.mark.asyncio
async def test_malformed_json_in_action_id_no_crash(orch_fixture: MagicMock) -> None:
    """Invalid JSON in action_id logs warning and returns gracefully."""
    from kiro_crew.slack import interactions

    payload, action, channel, msg_ts = _base_payload(value="", action_id="action::not{valid-json")
    action["selected_date"] = "2026-04-10"

    with patch.object(interactions, "handle_message", new_callable=AsyncMock) as mock_hm:
        # Should not raise
        await interactions._handle_options(payload, action, channel, msg_ts)
        mock_hm.assert_not_called()


@pytest.mark.asyncio
async def test_non_dict_json_in_action_id_no_crash(orch_fixture: MagicMock) -> None:
    """Non-dict JSON (e.g. a list) in action_id logs warning and returns."""
    from kiro_crew.slack import interactions

    payload, action, channel, msg_ts = _base_payload(value="", action_id='action::["a","b"]')
    action["selected_date"] = "2026-04-10"

    with patch.object(interactions, "handle_message", new_callable=AsyncMock) as mock_hm:
        await interactions._handle_options(payload, action, channel, msg_ts)
        mock_hm.assert_not_called()


@pytest.mark.asyncio
async def test_post_message_failure_aborts(orch_fixture: MagicMock) -> None:
    """If post_message returns None, handle_message is never called."""
    from kiro_crew.slack import interactions

    orch = orch_fixture
    orch.slack.post_message = AsyncMock(return_value=None)

    payload, action, channel, msg_ts = _base_payload(value='action::{"k":"v"}', action_id="btn_x")

    with patch.object(interactions, "handle_message", new_callable=AsyncMock) as mock_hm:
        await interactions._handle_options(payload, action, channel, msg_ts)
        await asyncio.sleep(0)
        mock_hm.assert_not_called()


@pytest.mark.asyncio
async def test_standard_options_post_message_failure_aborts(orch_fixture: MagicMock) -> None:
    """If update_message AND fallback post_blocks both fail, handle_message is never called."""
    from kiro_crew.slack import interactions

    orch = orch_fixture
    orch.slack.update_message = AsyncMock(side_effect=Exception("API error"))
    orch.slack.post_blocks = AsyncMock(return_value=None)

    payload, action, channel, msg_ts = _base_payload(value="some choice", action_id="opt_0")

    with patch.object(interactions, "handle_message", new_callable=AsyncMock) as mock_hm:
        await interactions._handle_options(payload, action, channel, msg_ts)
        mock_hm.assert_not_called()


@pytest.mark.asyncio
async def test_redaction_applied_to_payload(orch_fixture: MagicMock) -> None:
    """Exfiltration URLs in action payload are redacted before context."""
    from kiro_crew.slack import interactions

    orch = orch_fixture

    # base64-like blob (40+ chars) triggers _EXFIL_PATTERNS
    blob = "A" * 50
    evil_url = f"https://evil.com/exfil?data={blob}"
    payload, action, channel, msg_ts = _base_payload(
        value=f"action::{evil_url}", action_id="btn_evil"
    )

    with patch.object(interactions, "handle_message", new_callable=AsyncMock) as mock_hm:
        await interactions._handle_options(payload, action, channel, msg_ts)
        await asyncio.sleep(0)
        for t in list(orch._handler_tasks):
            await t

        ctx = mock_hm.call_args.kwargs["action_context"]
        # Full URL should be replaced with [REDACTED: ...]
        assert blob not in ctx
        assert "REDACTED" in ctx


@pytest.mark.asyncio
async def test_sel_audit_logged_for_action(orch_fixture: MagicMock) -> None:
    """SEL audit event is logged for action button interactions."""
    from kiro_crew.slack import interactions

    orch = orch_fixture

    payload, action, channel, msg_ts = _base_payload(
        value='action::{"k":"v"}', action_id="btn_audit"
    )

    with (
        patch.object(interactions, "handle_message", new_callable=AsyncMock),
        patch.object(interactions, "sel") as mock_sel,
    ):
        await interactions._handle_options(payload, action, channel, msg_ts)
        await asyncio.sleep(0)
        for t in list(orch._handler_tasks):
            await t

        mock_sel().log_api_access.assert_called_once()
        call_kwargs = mock_sel().log_api_access.call_args.kwargs
        assert call_kwargs["caller"] == "U123"
        assert call_kwargs["source"] == "slack"
        assert call_kwargs["outcome"] == "allowed"


# ---------------------------------------------------------------------------
# Phase 6 — stop_kill_now action and _handle_stop_confirm via stop_turn
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_slack_kill_now_action_force_stops(orch_fixture: MagicMock) -> None:
    """stop_kill_now action calls sessions.stop_turn with force=True."""
    from kiro_crew.slack import interactions
    from kiro_crew.slack.handler import set_allowed_users, set_owner_id

    set_owner_id("U123")
    set_allowed_users({"U123"})

    orch = orch_fixture
    orch.sessions.stop_turn = AsyncMock(return_value="hard")

    payload = {
        "type": "block_actions",
        "user": {"id": "U123"},
        "team": {"id": "T1"},
        "channel": {"id": "C1"},
        "response_url": "https://hooks.slack.com/actions/T1/fake",
        "message": {"ts": "200.0", "thread_ts": "100.0", "blocks": []},
        "actions": [
            {
                "action_id": "stop_kill_now",
                "value": "session_key_123",
                "text": {"type": "plain_text", "text": "Kill Now"},
            }
        ],
    }

    with patch.object(interactions, "sel") as mock_sel:
        mock_sel.return_value = MagicMock()
        await interactions.dispatch(payload)

    orch.sessions.stop_turn.assert_called_once()
    call_kwargs = orch.sessions.stop_turn.call_args
    assert call_kwargs.args[0] == "session_key_123"
    assert call_kwargs.kwargs["force"] is True


@pytest.mark.asyncio
async def test_slack_kill_now_posts_to_thread_not_session_key(
    orch_fixture: MagicMock,
) -> None:
    """stop_kill_now posts to the ephemeral's thread_ts, not to session_key.

    Regression test: for linked dashboard sessions, session_key is not a
    valid Slack thread (e.g. ``dashboard:chat-xxx``) and would fail to post.
    """
    from kiro_crew.slack import interactions
    from kiro_crew.slack.handler import set_allowed_users, set_owner_id

    set_owner_id("U123")
    set_allowed_users({"U123"})

    orch = orch_fixture
    # stop_turn must invoke on_hard for the post_message branch to run

    async def _fake_stop_turn(key, *, force=False, on_soft=None, on_hard=None):
        if on_hard:
            await on_hard()
        return "hard"

    orch.sessions.stop_turn = AsyncMock(side_effect=_fake_stop_turn)
    orch.slack = MagicMock()
    orch.slack.post_message = AsyncMock(return_value="300.0")

    payload = {
        "type": "block_actions",
        "user": {"id": "U123"},
        "team": {"id": "T1"},
        "channel": {"id": "C1"},
        "response_url": "https://hooks.slack.com/actions/T1/fake",
        "message": {"ts": "200.0", "thread_ts": "100.0", "blocks": []},
        "actions": [
            {
                "action_id": "stop_kill_now",
                "value": "dashboard:chat-xyz",  # linked-session key ≠ thread_ts
                "text": {"type": "plain_text", "text": "Kill Now"},
            }
        ],
    }

    # Mock aiohttp.ClientSession to prevent real HTTP call to the fake response_url;
    # _on_hard posts via aiohttp before calling orch.slack.post_message.
    mock_aio_session = AsyncMock()
    mock_aio_session.__aenter__ = AsyncMock(return_value=mock_aio_session)
    mock_aio_session.__aexit__ = AsyncMock(return_value=None)
    with (
        patch.object(interactions, "sel") as mock_sel,
        patch("aiohttp.ClientSession", return_value=mock_aio_session),
    ):
        mock_sel.return_value = MagicMock()
        await interactions.dispatch(payload)

    orch.slack.post_message.assert_called_once()
    args = orch.slack.post_message.call_args.args
    # (channel, text, thread_ts) — thread_ts must be 100.0, not dashboard:chat-xyz
    assert args[2] == "100.0"
    assert args[2] != "dashboard:chat-xyz"


@pytest.mark.asyncio
async def test_slack_kill_now_rejects_unauthorized(orch_fixture: MagicMock) -> None:
    """stop_kill_now enforces is_allowed_user() — deny-by-default."""
    from kiro_crew.slack import interactions

    orch = orch_fixture
    orch.sessions.stop_turn = AsyncMock(return_value="hard")

    payload = {
        "type": "block_actions",
        "user": {"id": "U_RANDOM"},  # Not allowlisted
        "team": {"id": "T1"},
        "channel": {"id": "C1"},
        "response_url": "https://hooks.slack.com/actions/T1/fake",
        "message": {"ts": "200.0", "thread_ts": "100.0", "blocks": []},
        "actions": [
            {
                "action_id": "stop_kill_now",
                "value": "session_key_123",
                "text": {"type": "plain_text", "text": "Kill Now"},
            }
        ],
    }

    with patch.object(interactions, "sel") as mock_sel:
        mock_sel.return_value = MagicMock()
        await interactions.dispatch(payload)

    # Unauthorized user — stop_turn must NOT be called
    orch.sessions.stop_turn.assert_not_called()


@pytest.mark.asyncio
async def test_handle_stop_kill_now_defense_in_depth(orch_fixture: MagicMock) -> None:
    """_handle_stop_kill_now re-checks is_allowed_user() even if dispatch() is bypassed.

    Defense-in-depth: if the dispatch() gate ever regresses (as happened
    with _handle_interactive), the handler must still deny unauthorized
    callers. Direct handler invocation simulates that bypass.
    """
    from kiro_crew.slack import interactions
    from kiro_crew.slack.handler import set_allowed_users, set_owner_id

    set_owner_id("U123")
    set_allowed_users({"U123"})

    orch = orch_fixture
    orch.sessions.stop_turn = AsyncMock(return_value="hard")

    payload = {"response_url": "", "message": {"ts": "200.0", "thread_ts": "100.0"}}
    action = {"action_id": "stop_kill_now", "value": "session_key_123"}

    with patch.object(interactions, "sel") as mock_sel:
        mock_sel.return_value = MagicMock()
        await interactions._handle_stop_kill_now(
            payload, action, channel="C1", msg_ts="200.0", user_id="U_RANDOM"
        )

    orch.sessions.stop_turn.assert_not_called()


@pytest.mark.asyncio
async def test_handle_stop_confirm_uses_stop_turn(orch_fixture: MagicMock) -> None:
    """/kirocrew stop confirm button routes through stop_turn, not bare reset."""
    from kiro_crew.slack import interactions
    from kiro_crew.slack.handler import set_allowed_users, set_owner_id

    set_owner_id("U123")
    set_allowed_users({"U123"})

    orch = orch_fixture
    orch.sessions.has_session = MagicMock(return_value=True)
    orch.sessions.stop_turn = AsyncMock(return_value="soft")
    orch._session_tasks = {}

    payload = {
        "type": "block_actions",
        "user": {"id": "U123"},
        "team": {"id": "T1"},
        "channel": {"id": "C1"},
        "response_url": "https://hooks.slack.com/actions/T1/fake",
        "message": {"ts": "200.0", "thread_ts": "100.0", "blocks": []},
        "actions": [
            {
                "action_id": "mc_stop_confirm",
                "value": "",
                "text": {"type": "plain_text", "text": "Confirm"},
            }
        ],
    }

    with patch.object(interactions, "sel") as mock_sel:
        mock_sel.return_value = MagicMock()
        await interactions.dispatch(payload)

    orch.sessions.stop_turn.assert_called_once()
    # Verify reset was NOT called directly
    orch.sessions.reset.assert_not_called()


@pytest.mark.asyncio
async def test_handle_stop_confirm_rejects_unauthorized(orch_fixture: MagicMock) -> None:
    """_handle_stop_confirm enforces is_allowed_user() — deny-by-default.

    stop_turn() can escalate to a hard kill, so the handler must re-check
    authorization even though dispatch() also enforces it.
    """
    from kiro_crew.slack import interactions
    from kiro_crew.slack.handler import set_allowed_users, set_owner_id

    set_owner_id("U123")
    set_allowed_users({"U123"})  # U_RANDOM not allowlisted

    orch = orch_fixture
    orch.sessions.has_session = MagicMock(return_value=True)
    orch.sessions.stop_turn = AsyncMock(return_value="soft")
    orch._session_tasks = {}

    payload = {
        "type": "block_actions",
        "user": {"id": "U_RANDOM"},
        "team": {"id": "T1"},
        "channel": {"id": "C1"},
        "response_url": "https://hooks.slack.com/actions/T1/fake",
        "message": {"ts": "200.0", "thread_ts": "100.0", "blocks": []},
        "actions": [
            {
                "action_id": "mc_stop_confirm",
                "value": "",
                "text": {"type": "plain_text", "text": "Confirm"},
            }
        ],
    }

    with patch.object(interactions, "sel") as mock_sel:
        mock_sel.return_value = MagicMock()
        await interactions.dispatch(payload)

    orch.sessions.stop_turn.assert_not_called()


@pytest.mark.asyncio
async def test_handle_stop_confirm_allows_roster_member(orch_fixture: MagicMock) -> None:
    """A prompt-roster non-admin may confirm a stop (member surface)."""
    from kiro_crew.slack import interactions
    from kiro_crew.slack.handler import set_allowed_users, set_owner_id

    set_owner_id("U_OWNER")
    set_allowed_users({"U_MEMBER"})

    orch = orch_fixture
    orch.sessions.has_session = MagicMock(return_value=True)
    orch.sessions.stop_turn = AsyncMock(return_value="soft")
    orch._session_tasks = {}

    payload = {
        "type": "block_actions",
        "user": {"id": "U_MEMBER"},
        "team": {"id": "T1"},
        "channel": {"id": "C1"},
        "response_url": "https://hooks.slack.com/actions/T1/fake",
        "message": {"ts": "200.0", "thread_ts": "100.0", "blocks": []},
        "actions": [
            {
                "action_id": "mc_stop_confirm",
                "value": "",
                "text": {"type": "plain_text", "text": "Confirm"},
            }
        ],
    }

    with patch.object(interactions, "sel") as mock_sel:
        mock_sel.return_value = MagicMock()
        await interactions.dispatch(payload)

    orch.sessions.stop_turn.assert_called_once()


@pytest.mark.asyncio
async def test_handle_stop_kill_now_allows_roster_member(orch_fixture: MagicMock) -> None:
    """A prompt-roster non-admin may use Kill Now (member surface)."""
    from kiro_crew.slack import interactions
    from kiro_crew.slack.handler import set_allowed_users, set_owner_id

    set_owner_id("U_OWNER")
    set_allowed_users({"U_MEMBER"})

    orch = orch_fixture
    orch.sessions.stop_turn = AsyncMock(return_value="hard")

    payload = {
        "type": "block_actions",
        "user": {"id": "U_MEMBER"},
        "team": {"id": "T1"},
        "channel": {"id": "C1"},
        "response_url": "https://hooks.slack.com/actions/T1/fake",
        "message": {"ts": "200.0", "thread_ts": "100.0", "blocks": []},
        "actions": [
            {
                "action_id": "stop_kill_now",
                "value": "session_key_123",
                "text": {"type": "plain_text", "text": "Kill Now"},
            }
        ],
    }

    with patch.object(interactions, "sel") as mock_sel:
        mock_sel.return_value = MagicMock()
        await interactions.dispatch(payload)

    orch.sessions.stop_turn.assert_called_once()


# ---------------------------------------------------------------------------
# Home Tab Resume — Slack views.publish payloads have empty channel +
# response_url, so _handle_session_resume must fall back to opening a DM
# with the user. Regression for the silent failure where the spinner
# resolved with no visible Resume confirmation.
# ---------------------------------------------------------------------------


class TestHomeTabSessionResume:
    def _orch_with_slack(self, *, has_link: tuple[str, str] = ("", "")):
        """Build a mock orchestrator + slack client. ``has_link`` controls
        whether the session is reported as already linked."""
        from kiro_crew.slack import interactions

        slack = MagicMock()
        slack.open_dm = AsyncMock(return_value="DUSER")
        slack.post_blocks = AsyncMock(return_value="100.0")
        slack.post_message = AsyncMock(return_value="100.0")

        sessions = MagicMock()
        sessions.get_slack_link = MagicMock(return_value=has_link)

        orch = MagicMock()
        orch.slack = slack
        orch.sessions = sessions
        orch.dashboard_state = MagicMock()
        # interactions._orch is a module-level cache that the dispatcher
        # consults, so swap it in.
        interactions._orch = orch
        return orch, slack, sessions

    def _home_tab_payload(self, key: str, title: str, user_id: str) -> dict:
        """Build a Slack ``block_actions`` payload as it arrives from a Home
        Tab button click — note no ``channel`` and no ``response_url``."""
        return {
            "type": "block_actions",
            "user": {"id": user_id},
            "team": {"id": "T1"},
            "container": {"type": "view", "view_id": "V1"},
            "view": {"type": "home", "id": "V1"},
            "actions": [
                {
                    "action_id": f"mc_session_resume_{key}",
                    "value": json.dumps({"key": key, "title": title}),
                }
            ],
        }

    @pytest.mark.asyncio
    async def test_falls_back_to_dm_when_channel_missing(self) -> None:
        from kiro_crew.slack import interactions

        orch, slack, sessions = self._orch_with_slack()
        payload = self._home_tab_payload("dashboard:chat-1", "Pipeline triage", "UOWNER")

        with (
            patch.object(interactions, "is_owner", return_value=True),
            patch.object(interactions, "is_allowed_user", return_value=True),
            patch.object(interactions, "sel") as mock_sel,
        ):
            mock_sel.return_value = MagicMock()
            await interactions.dispatch(payload)

        # Opens a DM with the clicker because the view payload has no channel
        slack.open_dm.assert_awaited_once_with("UOWNER")
        # Posts the Thread/DM choice blocks to the DM channel
        slack.post_blocks.assert_awaited_once()
        call_kwargs = slack.post_blocks.await_args
        # signature is post_blocks(channel, blocks, text)
        assert call_kwargs.args[0] == "DUSER"
        rendered = json.dumps(call_kwargs.args[1])
        # Both Thread and DM choice buttons present
        assert "mc_resume_thread_" in rendered
        assert "mc_resume_dm_" in rendered
        assert "Pipeline triage" in rendered

    @pytest.mark.asyncio
    async def test_already_linked_session_posts_message_to_dm(self) -> None:
        from kiro_crew.slack import interactions

        orch, slack, sessions = self._orch_with_slack(has_link=("123.456", "C0LINKED"))
        payload = self._home_tab_payload("dashboard:chat-1", "Live chat", "UOWNER")

        with (
            patch.object(interactions, "is_owner", return_value=True),
            patch.object(interactions, "is_allowed_user", return_value=True),
            patch.object(interactions, "sel") as mock_sel,
        ):
            mock_sel.return_value = MagicMock()
            await interactions.dispatch(payload)

        slack.open_dm.assert_awaited_once_with("UOWNER")
        slack.post_message.assert_awaited_once()
        # Goes to the DM channel and links to the existing thread
        args = slack.post_message.await_args.args
        assert args[0] == "DUSER"
        assert "C0LINKED" in args[1]
        assert "Go to conversation" in args[1]


# ---------------------------------------------------------------------------
# Transport tool-approval dispatch — deny-by-default authorization
# ---------------------------------------------------------------------------


class TestTransportApprovalAuth:
    """The messaging-transport approve/trust/deny buttons must never resolve a
    pending decider for an unauthorized clicker (deny-by-default)."""

    def _payload(self, action_id: str, user_id: str) -> dict:
        return {
            "type": "block_actions",
            "user": {"id": user_id},
            "channel": {"id": "C1"},
            "message": {"ts": "200.0"},
            "actions": [{"action_id": action_id, "value": "rq1", "text": {"text": "x"}}],
        }

    @pytest.mark.asyncio
    async def test_unauthorized_click_does_not_resolve(
        self, orch_fixture: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from kiro_crew.slack import interactions
        from kiro_crew.slack.renderer import SlackApprovalDecider

        monkeypatch.setattr(interactions, "is_allowed_user", lambda uid: False)
        spy = MagicMock(return_value=True)
        monkeypatch.setattr(SlackApprovalDecider, "resolve_global", spy)

        for action_id in ("mc_tool_approve_rq1", "mc_tool_trust_rq1", "mc_tool_deny_rq1"):
            await interactions.dispatch(self._payload(action_id, "U_INTRUDER"))
        # Never resolved the pending approval for an unauthorized user.
        spy.assert_not_called()

    @pytest.mark.asyncio
    async def test_authorized_approve_resolves(
        self, orch_fixture: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from kiro_crew.slack import interactions
        from kiro_crew.slack.renderer import SlackApprovalDecider

        monkeypatch.setattr(interactions, "is_allowed_user", lambda uid: True)
        spy = MagicMock(return_value=True)
        monkeypatch.setattr(SlackApprovalDecider, "resolve_global", spy)
        monkeypatch.setattr(
            SlackApprovalDecider, "match_failure_reason", MagicMock(return_value=None)
        )

        await interactions.dispatch(self._payload("mc_tool_approve_rq1", "U_OWNER"))
        spy.assert_called_once_with("rq1", True)
        orch_fixture.slack.update_message.assert_awaited()

    @pytest.mark.asyncio
    async def test_channels_deny_drops_transport_approval(
        self, orch_fixture: MagicMock, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        # HIGH (GPT round-8) + MEDIUM (GPT round-13 #3): an AUTHORIZED user's
        # APPROVE click on the transport path must not EXECUTE the governed tool
        # when a channels policy denies slack. Originally the gate silently returned
        # without resolving — but that STRANDS the kiro-cli approval future until it
        # times out (~300s). The fix resolves the pending future as DENIED (False)
        # so the tool is refused promptly and never executes. Assert the resolve was
        # a denial, not that it never happened.
        import json

        from kiro_crew.platform import governance_profiles as gp
        from kiro_crew.slack import interactions
        from kiro_crew.slack.renderer import SlackApprovalDecider

        pdir = tmp_path / "profiles"
        pdir.mkdir()
        monkeypatch.setattr(gp, "_PROFILES_DIR", pdir)
        gp.reset_store()
        (pdir / "host.json").write_text(
            json.dumps(
                {
                    "name": "host",
                    "bind": {"type": "surface", "id": "host"},
                    "channels": {"members": {"mode": "allow", "allow": ["discord"]}},
                }
            )
        )
        monkeypatch.setattr(interactions, "is_allowed_user", lambda uid: True)
        spy = MagicMock(return_value=True)
        monkeypatch.setattr(SlackApprovalDecider, "resolve_global", spy)
        monkeypatch.setattr(
            SlackApprovalDecider, "match_failure_reason", MagicMock(return_value=None)
        )
        try:
            await interactions.dispatch(self._payload("mc_tool_approve_rq1", "U_OWNER"))
            # Denied by the channels gate → the approval is resolved as DENIED
            # (False), never as approved (True), so the tool is refused, not run,
            # and the pending future is not left to time out.
            assert spy.call_count == 1
            _args, _kwargs = spy.call_args
            approved_arg = _kwargs.get("approved", _args[1] if len(_args) > 1 else None)
            assert approved_arg is False, "denied-channel approve must resolve as False (refused)"
        finally:
            gp.reset_store()

    @pytest.mark.asyncio
    async def test_channels_deny_still_resolves_transport_reject(
        self, orch_fixture: MagicMock, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        # MEDIUM (GPT round-13 #3): a REJECT click on a denied channel must STILL
        # resolve the pending approval as refused (False) — a reject is a denial,
        # exactly what a channels-deny wants, and dropping it would strand the
        # kiro-cli future until timeout. The reject is NOT gated out.
        import json

        from kiro_crew.platform import governance_profiles as gp
        from kiro_crew.slack import interactions
        from kiro_crew.slack.renderer import SlackApprovalDecider

        pdir = tmp_path / "profiles"
        pdir.mkdir()
        monkeypatch.setattr(gp, "_PROFILES_DIR", pdir)
        gp.reset_store()
        (pdir / "host.json").write_text(
            json.dumps(
                {
                    "name": "host",
                    "bind": {"type": "surface", "id": "host"},
                    "channels": {"members": {"mode": "allow", "allow": ["discord"]}},
                }
            )
        )
        monkeypatch.setattr(interactions, "is_allowed_user", lambda uid: True)
        spy = MagicMock(return_value=True)
        monkeypatch.setattr(SlackApprovalDecider, "resolve_global", spy)
        monkeypatch.setattr(
            SlackApprovalDecider, "match_failure_reason", MagicMock(return_value=None)
        )
        try:
            await interactions.dispatch(self._payload("mc_tool_deny_rq1", "U_OWNER"))
            assert spy.call_count == 1
            _args, _kwargs = spy.call_args
            approved_arg = _kwargs.get("approved", _args[1] if len(_args) > 1 else None)
            assert approved_arg is False, "a reject must resolve the pending approval as False"
        finally:
            gp.reset_store()

    @pytest.mark.asyncio
    async def test_channels_deny_drops_review_approve(
        self, orch_fixture: MagicMock, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        # HIGH (GPT round-9 #3): a review-mode APPROVE posts the stored agent draft
        # to the channel. Under a slack channels deny, the dispatch gate must stop
        # it BEFORE the handler runs — else stale agent content posts to a denied
        # channel. Regression-locks the review-action call site.
        import json

        from kiro_crew.platform import governance_profiles as gp
        from kiro_crew.slack import interactions

        pdir = tmp_path / "profiles"
        pdir.mkdir()
        monkeypatch.setattr(gp, "_PROFILES_DIR", pdir)
        gp.reset_store()
        (pdir / "host.json").write_text(
            json.dumps(
                {
                    "name": "host",
                    "bind": {"type": "surface", "id": "host"},
                    "channels": {"members": {"mode": "allow", "allow": ["discord"]}},
                }
            )
        )
        monkeypatch.setattr(interactions, "is_allowed_user", lambda uid: True)
        handler = AsyncMock()
        monkeypatch.setattr(interactions, "_handle_review_approve", handler)
        try:
            await interactions.dispatch(self._payload("mc_review_approve", "U_OWNER"))
            # Gate dropped it before the posting handler ran.
            handler.assert_not_called()
        finally:
            gp.reset_store()

    @pytest.mark.asyncio
    async def test_channels_deny_gates_native_tool_approval_but_not_reject(
        self, orch_fixture: MagicMock, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        # MEDIUM (fleet review): the NATIVE _handle_tool_approval gate (approve/trust
        # blocked, reject exempt) had no test, unlike the transport/Discord/Telegram
        # paths. Under a slack channels deny: an APPROVE must be gated BEFORE
        # handle_interaction runs (the governed tool never executes), but a REJECT
        # must still REACH handle_interaction (a reject is a denial — dropping it
        # would strand the pending approval until timeout).
        import json

        from kiro_crew.platform import governance_profiles as gp
        from kiro_crew.slack import interactions

        pdir = tmp_path / "profiles"
        pdir.mkdir()
        monkeypatch.setattr(gp, "_PROFILES_DIR", pdir)
        gp.reset_store()
        (pdir / "host.json").write_text(
            json.dumps(
                {
                    "name": "host",
                    "bind": {"type": "surface", "id": "host"},
                    "channels": {"members": {"mode": "allow", "allow": ["discord"]}},
                }
            )
        )
        hi = AsyncMock(return_value="approve_tool")
        monkeypatch.setattr(interactions, "handle_interaction", hi)
        try:
            # APPROVE on a denied channel → gated before handle_interaction.
            await interactions._handle_tool_approval(
                {"message": {}}, "approve_tool", "C1", "200.0", "U_OWNER"
            )
            hi.assert_not_called()
            # REJECT on the same denied channel → still reaches handle_interaction
            # (resolves the pending approval as refused, not stranded).
            await interactions._handle_tool_approval(
                {"message": {}}, "reject_tool", "C1", "200.0", "U_OWNER"
            )
            hi.assert_awaited_once()
        finally:
            gp.reset_store()

    @pytest.mark.asyncio
    async def test_channels_deny_drops_dashboard_link(
        self, orch_fixture: MagicMock, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        # BLOCKING (GPT round-17): the Link-to-Dashboard button imports the Slack
        # thread's content into a dashboard slot (_import_thread_to_slot). A channels
        # policy denying slack must stop it BEFORE the import — else a stale link
        # button moves denied Slack content into the dashboard. Regression-locks the
        # LINK_DASHBOARD_ACTION call site.
        import json

        from kiro_crew.platform import governance_profiles as gp
        from kiro_crew.slack import interactions

        pdir = tmp_path / "profiles"
        pdir.mkdir()
        monkeypatch.setattr(gp, "_PROFILES_DIR", pdir)
        gp.reset_store()
        (pdir / "host.json").write_text(
            json.dumps(
                {
                    "name": "host",
                    "bind": {"type": "surface", "id": "host"},
                    "channels": {"members": {"mode": "allow", "allow": ["discord"]}},
                }
            )
        )
        monkeypatch.setattr(interactions, "is_allowed_user", lambda uid: True)
        importer = AsyncMock()
        monkeypatch.setattr(interactions, "_import_thread_to_slot", importer)
        try:
            payload = self._payload("mc_link_dashboard", "U_OWNER")
            payload["message"]["thread_ts"] = "200.0"  # a thread to import
            await interactions.dispatch(payload)
            # Gate dropped it before the content-importing handler ran.
            importer.assert_not_called()
        finally:
            gp.reset_store()

    @pytest.mark.asyncio
    async def test_authorized_trust_grants_session_then_resolves(
        self, orch_fixture: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from kiro_crew.slack import interactions
        from kiro_crew.slack.renderer import SlackApprovalDecider

        monkeypatch.setattr(interactions, "is_allowed_user", lambda uid: True)
        monkeypatch.setattr(interactions, "is_prompt_allowed_user", lambda uid: True)
        monkeypatch.setattr(
            SlackApprovalDecider, "session_for", classmethod(lambda cls, rid: "thread-1")
        )
        monkeypatch.setattr(SlackApprovalDecider, "resolve_global", MagicMock(return_value=True))
        monkeypatch.setattr(
            SlackApprovalDecider, "match_failure_reason", MagicMock(return_value=None)
        )
        grant = MagicMock()
        monkeypatch.setattr(interactions, "add_trusted_session", grant)

        await interactions.dispatch(self._payload("mc_tool_trust_rq1", "U_OWNER"))
        # Trust granted for the resolved session before the approval resolves.
        grant.assert_called_once()
        assert grant.call_args.args[0] == "thread-1"


class TestToolApprovalWithoutOrchestrator:
    @pytest.mark.asyncio
    async def test_reject_reaches_handler_before_orchestrator_is_ready(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A reject must unblock the pending ACP future even during startup."""
        import kiro_crew.slack.interactions as interactions

        handler = AsyncMock(return_value="reject_tool")
        monkeypatch.setattr(interactions, "_orch", None)
        monkeypatch.setattr(interactions, "handle_interaction", handler)

        response = MagicMock(status=200)
        session = MagicMock()
        session.post = AsyncMock(return_value=response)
        client = MagicMock()
        client.__aenter__ = AsyncMock(return_value=session)
        with patch.object(interactions.aiohttp, "ClientSession", return_value=client):
            await interactions._handle_tool_approval(
                {
                    "response_url": "https://hooks.slack.com/T/B/token",
                    "message": {
                        "blocks": [
                            {"type": "section", "text": {"type": "mrkdwn", "text": "Approve?"}},
                            {"type": "actions", "elements": [{"action_id": "reject_tool"}]},
                        ]
                    },
                },
                "reject_tool",
                "D1",
                "approval",
                "U_OWNER",
            )

        handler.assert_awaited_once_with(
            "D1",
            "approval",
            "reject_tool",
            user_id="U_OWNER",
            thread_ts="",
            slack=None,
            sessions=None,
        )
        update = session.post.await_args.kwargs["json"]
        assert update["replace_original"] is True
        assert update["text"] == "🚫 Rejected"
        assert all(block["type"] != "actions" for block in update["blocks"])


class TestToolApprovalAttribution:
    """Task 4b: every resolved tool approval (native or linked; approve,
    reject, or trust) posts one threaded message naming the clicker."""

    def _payload(self, thread_ts: str = "root") -> dict:
        return {
            "message": {"ts": "m1", "thread_ts": thread_ts, "blocks": []},
        }

    @pytest.mark.asyncio
    async def test_approve_posts_attribution_with_mention_and_check(
        self,
        orch_fixture: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from kiro_crew.slack import interactions

        monkeypatch.setattr(
            interactions, "handle_interaction", AsyncMock(return_value="approve_tool")
        )

        with caplog.at_level(logging.INFO, logger=interactions.__name__):
            await interactions._handle_tool_approval(
                self._payload(thread_ts="root"), "approve_tool", "C1", "m1", "U_MEMBER"
            )

        orch_fixture.slack.post_message.assert_awaited_once()
        call = orch_fixture.slack.post_message.await_args
        assert call.args[0] == "C1"
        assert "<@U_MEMBER>" in call.args[1]
        assert "✅" in call.args[1]
        assert call.kwargs["thread_ts"] == "root"
        assert "Tool approval resolved: action=approve_tool by=U_MEMBER" in caplog.text

    @pytest.mark.asyncio
    async def test_reject_posts_attribution_with_mention(
        self, orch_fixture: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from kiro_crew.slack import interactions

        monkeypatch.setattr(
            interactions, "handle_interaction", AsyncMock(return_value="reject_tool")
        )

        await interactions._handle_tool_approval(
            self._payload(thread_ts="root"), "reject_tool", "C1", "m1", "U_MEMBER"
        )

        orch_fixture.slack.post_message.assert_awaited_once()
        text = orch_fixture.slack.post_message.await_args.args[1]
        assert "🚫" in text
        assert "<@U_MEMBER>" in text

    @pytest.mark.asyncio
    async def test_trust_posts_attribution_mentioning_trust(
        self, orch_fixture: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from kiro_crew.slack import interactions

        monkeypatch.setattr(
            interactions, "handle_interaction", AsyncMock(return_value="trust_tool")
        )

        await interactions._handle_tool_approval(
            self._payload(thread_ts="root"), "trust_tool", "C1", "m1", "U_OWNER"
        )

        orch_fixture.slack.post_message.assert_awaited_once()
        text = orch_fixture.slack.post_message.await_args.args[1]
        assert "trust" in text.lower()
        assert "<@U_OWNER>" in text

    @pytest.mark.asyncio
    async def test_unresolved_click_does_not_post_attribution(
        self, orch_fixture: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """effective_action is None => ephemeral-only path (Task 4), never
        double-notify with a public attribution message."""
        from kiro_crew.slack import interactions

        monkeypatch.setattr(interactions, "handle_interaction", AsyncMock(return_value=None))

        await interactions._handle_tool_approval(
            self._payload(thread_ts="root"), "approve_tool", "C1", "m1", "U_MEMBER"
        )

        orch_fixture.slack.post_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_thread_ts_posts_attribution_top_level(
        self, orch_fixture: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from kiro_crew.slack import interactions

        monkeypatch.setattr(
            interactions, "handle_interaction", AsyncMock(return_value="approve_tool")
        )

        await interactions._handle_tool_approval(
            self._payload(thread_ts=""), "approve_tool", "C1", "m1", "U_MEMBER"
        )

        assert orch_fixture.slack.post_message.await_args.kwargs["thread_ts"] is None

    @pytest.mark.asyncio
    async def test_post_message_failure_does_not_raise(
        self, orch_fixture: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from kiro_crew.slack import interactions

        monkeypatch.setattr(
            interactions, "handle_interaction", AsyncMock(return_value="approve_tool")
        )
        orch_fixture.slack.post_message = AsyncMock(side_effect=RuntimeError("slack down"))

        await interactions._handle_tool_approval(
            self._payload(thread_ts="root"), "approve_tool", "C1", "m1", "U_MEMBER"
        )

        # No exception propagated (the await above completed), and the
        # attribution post was still attempted despite Slack raising.
        orch_fixture.slack.post_message.assert_awaited_once()
