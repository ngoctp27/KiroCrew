"""Tests for Slack-linked dashboard-slot tool approvals.

When a dashboard slot is linked to a Slack thread and driven from Slack, the
tool-approval prompt must be mirrored into that Slack thread so the user can
answer it. A Slack button click resolves the *dashboard slot's* approval future
(via ``state.resolve_approval``) and must NOT answer the ACP backend directly —
the dashboard ``_run_chat`` loop remains the sole caller of approve/reject.

Regression coverage for the "no approval request for linked sessions" bug.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import kiro_crew.slack.handler as handler
from kiro_crew.slack.handler import (
    _ACTION_APPROVE,
    _ACTION_REJECT,
    _linked_approvals,
    handle_interaction,
    post_linked_approval,
    resolve_linked_approval,
    set_admin_users,
    set_allowed_users,
    set_owner_id,
)


@pytest.fixture(autouse=True)
def _clear_registry():
    """Each test gets isolated approval and Slack authorization state."""
    _linked_approvals.clear()
    handler._pending_approvals.clear()
    set_owner_id("U_OWNER")
    set_allowed_users({"U_MEMBER"})
    yield
    _linked_approvals.clear()
    handler._pending_approvals.clear()
    set_owner_id("")


def _make_slack(post_ts: str | None = "1781300000.0001") -> MagicMock:
    slack = MagicMock()
    if post_ts is None:
        slack.post_blocks = AsyncMock(side_effect=RuntimeError("slack down"))
    else:
        slack.post_blocks = AsyncMock(return_value=post_ts)
    slack.delete_message = AsyncMock()
    return slack


# ── post_linked_approval ───────────────────────────────────────────────────


class TestPostLinkedApproval:
    @pytest.mark.asyncio
    async def test_posts_buttons_and_registers_entry(self) -> None:
        slack = _make_slack("1781300000.0001")
        ts = await post_linked_approval(
            slack,
            channel="C_LINK",
            thread_ts="1781290000.0001",
            request_id=42,
            session_key="dashboard:chat-1-123",
            title="shell: ls -la",
            tool_input="ls -la",
        )
        assert ts == "1781300000.0001"
        slack.post_blocks.assert_awaited_once()
        # Posted threaded under the linked thread, to the linked channel.
        args = slack.post_blocks.call_args.args
        assert args[0] == "C_LINK"
        assert args[3] == "1781290000.0001"
        # Registry entry created, keyed by channel:ts, carrying the request id.
        entry = _linked_approvals["C_LINK:1781300000.0001"]
        assert entry.request_id == 42
        assert entry.session_key == "dashboard:chat-1-123"

    @pytest.mark.asyncio
    async def test_delivery_failure_returns_none_no_entry(self) -> None:
        """A failed Slack post returns None and registers nothing — the caller
        treats None as 'delivery failed' rather than parking forever."""
        slack = _make_slack(post_ts=None)
        ts = await post_linked_approval(
            slack,
            channel="C_LINK",
            thread_ts="1781290000.0001",
            request_id=7,
            session_key="dashboard:chat-1-123",
            title="shell: rm -rf /tmp/x",
            tool_input="rm -rf /tmp/x",
        )
        assert ts is None
        assert _linked_approvals == {}

    @pytest.mark.asyncio
    async def test_no_trust_button_on_linked_path(self) -> None:
        """Linked prompts offer Approve/Reject only (Trust is dashboard-side)."""
        slack = _make_slack("1781300000.0001")
        with patch.object(
            handler, "_build_approval_blocks", wraps=handler._build_approval_blocks
        ) as spy:
            await post_linked_approval(
                slack,
                channel="C_LINK",
                thread_ts="1781290000.0001",
                request_id=1,
                session_key="dashboard:chat-1-123",
                title="shell: ls",
                tool_input="ls",
            )
        assert spy.call_args.kwargs.get("allow_trust") is False

    @pytest.mark.asyncio
    async def test_redacts_llm_output_before_posting(self) -> None:
        """title / tool_input are LLM-generated; both must be scrubbed with
        redact_exfiltration_urls AND redact_credentials before reaching Slack
        (Slack is an external surface)."""
        slack = _make_slack("1781300000.0001")
        with patch.object(
            handler, "redact_exfiltration_urls", side_effect=lambda s: (s, [])
        ) as exfil, patch.object(
            handler, "redact_credentials", side_effect=lambda s: (s, [])
        ) as cred:
            await post_linked_approval(
                slack,
                channel="C_LINK",
                thread_ts="1781290000.0001",
                request_id=5,
                session_key="dashboard:chat-1-123",
                title="shell: curl http://x",
                tool_input="curl http://x --header secret",
            )
        # Both redactors ran over both LLM-generated strings.
        exfil_inputs = [c.args[0] for c in exfil.call_args_list]
        cred_inputs = [c.args[0] for c in cred.call_args_list]
        assert "shell: curl http://x" in exfil_inputs
        assert "curl http://x --header secret" in exfil_inputs
        assert "shell: curl http://x" in cred_inputs
        assert "curl http://x --header secret" in cred_inputs


# ── handle_interaction: linked-slot routing ────────────────────────────────


class TestLinkedInteractionRouting:
    def _arm(self, request_id: str = "99") -> None:
        _linked_approvals["C_LINK:TS1"] = handler._LinkedApproval(
            request_id=request_id, session_key="dashboard:chat-1-123"
        )

    @pytest.mark.asyncio
    async def test_approve_resolves_future_not_backend(self) -> None:
        """Approve click resolves the slot future via resolve_approval and does
        NOT call approve_tool/reject_tool (the dashboard loop owns that)."""
        self._arm("99")
        dstate = MagicMock()
        dstate.resolve_approval = MagicMock(return_value=True)
        with patch.object(handler, "_dashboard_state", dstate):
            result = await handle_interaction(
                "C_LINK", "TS1", _ACTION_APPROVE, user_id="U_OWNER"
            )
        assert result == _ACTION_APPROVE
        dstate.resolve_approval.assert_called_once_with("99", True)
        # Registry entry consumed.
        assert "C_LINK:TS1" not in _linked_approvals

    @pytest.mark.asyncio
    async def test_reject_resolves_future_with_false(self) -> None:
        self._arm("99")
        dstate = MagicMock()
        dstate.resolve_approval = MagicMock(return_value=True)
        with patch.object(handler, "_dashboard_state", dstate):
            result = await handle_interaction(
                "C_LINK", "TS1", _ACTION_REJECT, user_id="U_OWNER"
            )
        assert result == _ACTION_REJECT
        dstate.resolve_approval.assert_called_once_with("99", False)
        assert "C_LINK:TS1" not in _linked_approvals

    @pytest.mark.asyncio
    async def test_unauthorized_user_rejected_before_resolve(self) -> None:
        """A non-allowed user must not resolve a linked approval."""
        self._arm("99")
        dstate = MagicMock()
        dstate.resolve_approval = MagicMock(return_value=True)
        with patch.object(handler, "_dashboard_state", dstate):
            result = await handle_interaction(
                "C_LINK", "TS1", _ACTION_APPROVE, user_id="U_STRANGER"
            )
        assert result is None
        dstate.resolve_approval.assert_not_called()
        # Entry preserved so the owner can still act.
        assert "C_LINK:TS1" in _linked_approvals

    @pytest.mark.asyncio
    @pytest.mark.parametrize("action_id", [_ACTION_APPROVE, _ACTION_REJECT])
    async def test_roster_member_cannot_resolve_linked_approval(
        self, action_id: str
    ) -> None:
        """A roster member (not owner, not admin) must NOT be able to resolve
        a linked approval: tool approval is an owner/admin-only decision on
        BOTH the native and linked paths, and a linked slot carries no Slack
        requester identity to make an exception for. The member's click must
        be denied before ever touching the dashboard slot future or the ACP
        provider.

        A same-keyed _PendingApproval carrying the provider mock is
        registered below so the "never touches the provider" claim is
        actually falsifiable rather than vacuous.

        RED until Task 5 changes the top-of-function owner/admin gate to is_owner.
        """
        self._arm("99")
        dstate = MagicMock()
        dstate.resolve_approval = MagicMock(return_value=True)
        provider = MagicMock()
        provider.approve_tool = AsyncMock()
        provider.reject_tool = AsyncMock()
        # reply_ts must be "" here, not "TS1": handle_interaction is called
        # below with no thread_ts (defaults to ""), and
        # _approval_session_matches("TS1", "", "TS1") would hit
        # `if not thread_ts: return False` (session mismatch) and return
        # None before ever reaching the provider -- making the
        # assert_not_awaited() below pass vacuously regardless of which gate
        # denies the click. An empty reply_ts hits the empty-reply_ts branch
        # instead, so a None result here is provably the roster gate at the
        # top of handle_interaction, not an unrelated session mismatch on
        # the native path this same key would otherwise fall through to.
        handler._pending_approvals["C_LINK:TS1"] = handler._PendingApproval(
            provider=provider,
            request_id="99",
            session_key="dashboard:chat-1-123",
            requester_id="U_MEMBER",
            reply_ts="",
        )
        with patch.object(handler, "_dashboard_state", dstate):
            result = await handle_interaction(
                "C_LINK", "TS1", action_id, user_id="U_MEMBER"
            )
        assert result is None
        dstate.resolve_approval.assert_not_called()
        # Positive assert: the entry survives, proving this was a denial
        # (nothing consumed or resolved the linked slot) rather than a
        # successful resolve. The top-of-function owner/admin gate runs BEFORE the
        # linked-registry read, so this does not prove the code
        # reached that read -- dstate.resolve_approval.assert_not_called()
        # above is what rules out the linked branch having run to
        # completion; this assert only rules out anything having deleted
        # the entry.
        assert "C_LINK:TS1" in _linked_approvals
        provider.approve_tool.assert_not_awaited()
        provider.reject_tool.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_admin_can_resolve_linked_approval(self) -> None:
        """An additional admin (not the primary owner, not on the member
        roster) must also be able to resolve a linked approval.

        Green today via the top-of-function owner/admin gate
        (is_prompt_allowed_user's is_owner() branch) --
        the linked branch itself carries no separate authority check.
        After Task 5 changes that gate's predicate to is_owner, this
        continues to pass through the same gate for the same reason: an
        admin is admitted at the top of handle_interaction and never needs
        to touch the requester exception at the requester-match bypass, which exists
        for the native path only -- the linked branch has no requester
        identity to bind to in the first place.
        """
        set_admin_users({"U_ADMIN2"})
        self._arm("99")
        dstate = MagicMock()
        dstate.resolve_approval = MagicMock(return_value=True)
        with patch.object(handler, "_dashboard_state", dstate):
            result = await handle_interaction(
                "C_LINK", "TS1", _ACTION_APPROVE, user_id="U_ADMIN2"
            )
        assert result == _ACTION_APPROVE
        dstate.resolve_approval.assert_called_once_with("99", True)
        assert "C_LINK:TS1" not in _linked_approvals

    @pytest.mark.asyncio
    async def test_owner_can_resolve_linked_approval(self) -> None:
        """The primary owner must also be able to resolve a linked approval.

        Distinct from test_approve_resolves_future_not_backend /
        test_reject_resolves_future_with_false above: those pin the
        "resolves the dashboard future, never the ACP provider directly"
        invariant. This one pins owner authority specifically, in the same
        shape as test_admin_can_resolve_linked_approval, so the two admit
        paths (primary owner vs. KIROCREW_ADMIN_USER_IDS) both have a
        dedicated case rather than owner coverage only being incidental to
        an unrelated test.
        """
        self._arm("99")
        dstate = MagicMock()
        dstate.resolve_approval = MagicMock(return_value=True)
        with patch.object(handler, "_dashboard_state", dstate):
            result = await handle_interaction(
                "C_LINK", "TS1", _ACTION_APPROVE, user_id="U_OWNER"
            )
        assert result == _ACTION_APPROVE
        dstate.resolve_approval.assert_called_once_with("99", True)
        assert "C_LINK:TS1" not in _linked_approvals

    @pytest.mark.asyncio
    async def test_non_roster_user_denied_and_entry_preserved(self) -> None:
        """A user outside the roster entirely (not owner, not admin, not on
        the member allowlist) must still be denied, and the linked entry
        must be preserved so a roster member can still act on it.

        Distinct from test_unauthorized_user_rejected_before_resolve: this
        case additionally pins the SEL denial audit fields (spec Success
        Criterion 2), so a future change that silences or mislabels the
        audit trail for this gate is caught here."""
        self._arm("99")
        dstate = MagicMock()
        dstate.resolve_approval = MagicMock(return_value=True)
        mock_sel = MagicMock()
        with patch.object(handler, "_dashboard_state", dstate), patch.object(
            handler, "sel", return_value=mock_sel
        ):
            result = await handle_interaction(
                "C_LINK", "TS1", _ACTION_APPROVE, user_id="U_STRANGER"
            )
        assert result is None
        dstate.resolve_approval.assert_not_called()
        assert "C_LINK:TS1" in _linked_approvals
        mock_sel.log_api_access.assert_called_once_with(
            caller="U_STRANGER",
            operation="slack.interactive.approval",
            outcome="denied",
            source="slack",
            resources=_ACTION_APPROVE,
            error="unauthorized user",
        )

    @pytest.mark.asyncio
    async def test_roster_revocation_fails_closed(self) -> None:
        """Once the member roster is revoked (set_allowed_users(set())), a
        formerly-allowlisted member must be denied -- fails closed. This is
        enforced by the single owner/admin gate at the top of handle_interaction.
        After Task 5 changes that gate's predicate to
        is_owner, a revoked member is denied for the same underlying reason
        (no longer an admit path), since set_allowed_users only ever
        controlled is_prompt_allowed_user, not is_owner."""
        set_allowed_users(set())
        self._arm("99")
        dstate = MagicMock()
        dstate.resolve_approval = MagicMock(return_value=True)
        with patch.object(handler, "_dashboard_state", dstate):
            result = await handle_interaction(
                "C_LINK", "TS1", _ACTION_APPROVE, user_id="U_MEMBER"
            )
        assert result is None
        dstate.resolve_approval.assert_not_called()
        assert "C_LINK:TS1" in _linked_approvals

    @pytest.mark.asyncio
    async def test_linked_entry_takes_precedence_over_pending(self) -> None:
        """A linked entry must be handled before the _pending_approvals path so
        a linked click never double-answers the backend."""
        self._arm("99")
        dstate = MagicMock()
        dstate.resolve_approval = MagicMock(return_value=True)
        # Also place a (bogus) pending approval under the same key — the linked
        # branch must win and never touch it.
        with patch.object(handler, "_dashboard_state", dstate), patch.dict(
            handler._pending_approvals, {}, clear=False
        ):
            result = await handle_interaction(
                "C_LINK", "TS1", _ACTION_APPROVE, user_id="U_OWNER"
            )
        assert result == _ACTION_APPROVE
        dstate.resolve_approval.assert_called_once()


# ── resolve_linked_approval ────────────────────────────────────────────────


def test_resolve_linked_approval_drops_entry() -> None:
    _linked_approvals["C_LINK:TS1"] = handler._LinkedApproval(
        request_id="1", session_key="dashboard:chat-1-123"
    )
    resolve_linked_approval("C_LINK", "TS1")
    assert "C_LINK:TS1" not in _linked_approvals
    # Idempotent — second call is a no-op.
    resolve_linked_approval("C_LINK", "TS1")
