"""Activation regressions for the semantic reply-policy + deterministic privacy gate (#921).

These exercise the REAL production seams — no stand-in runners:

  * ``BasePlatformAdapter._reply_policy_for`` / the WhatsApp override actually
    stamp ``SessionSource.reply_policy`` at ingress, and the real gate predicate
    ``GatewayTurnMixin._is_reply_tool_gated`` arms only for a free-response group
    turn under ``reply_gate_mode="tool"``.
  * ``GatewayRunner._hmwa_deliver_turn_response`` — the single seam every final
    delivery lane (direct completion, queued follow-up, streamed turn) funnels
    through — suppresses an owner-mechanism privacy leak to ``""`` before the
    adapter is ever selected, so no leak reaches the wire on ANY lane.

The owner-name alternation the deterministic gate keys on is derived from
``HERMES_OWNER_NAMES`` at import; the neutral default is ``morgan``, so the leak
fixtures below name ``morgan`` and carry a distinct infra signal (two signals =
held), which is exactly the co-occurrence the production gate requires.
"""

import pytest

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType, SessionSource
from gateway.reply_policy import ReplyPolicy, resolve_reply_policy
from gateway.run_turn import GatewayTurnMixin
from tests.gateway.test_run_progress_topics import ProgressCaptureAdapter, _make_runner

# A message that trips BOTH deterministic signals with the default owner token
# ("morgan"): owner-reference ("asistente de morgan") + infra/model reference
# ("por debajo corre un modelo que él configura ... alias"). This is the exact
# owner-mechanism disclosure class the gate exists to hold.
_LEAK = (
    "Sí, soy el asistente de morgan; por debajo corre un modelo que él "
    "configura y lo tiene bajo el alias 'x'."
)
_CLEAN = "Listo, ya te mandé el resumen. ¿Necesitas algo más?"

_SESSION_KEY = "agent:main:telegram:group:-1001:17585"


# --------------------------------------------------------------------------- #
# reply_policy stamping + gate arming (real adapter + real predicate)
# --------------------------------------------------------------------------- #
class _StubAdapter(BasePlatformAdapter):
    """Minimal concrete adapter to exercise base ``_reply_policy_for`` directly."""

    def __init__(self, *, free_response=(), require_mention=True):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)
        self._free = set(free_response)
        self._require = require_mention

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise AssertionError("send must not be called by these tests")

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}

    # Base hooks the ingress stamping consults.
    def _adapter_require_mention(self) -> bool:
        return self._require

    def _chat_in_free_response_set(self, chat_id: str) -> bool:
        return str(chat_id) in self._free


def _gate_arms(reply_policy, mode="tool"):
    """Call the REAL production predicate every delivery seam consults."""
    from types import SimpleNamespace

    obj = object.__new__(GatewayTurnMixin)
    obj.config = SimpleNamespace(reply_gate_mode=mode)
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="-1001", chat_type="group",
        reply_policy=reply_policy,
    )
    return GatewayTurnMixin._is_reply_tool_gated(obj, source)


class TestReplyPolicyStamping:
    def test_dm_is_dm_policy(self):
        adapter = _StubAdapter()
        assert adapter._reply_policy_for("private", "u1") == ReplyPolicy.DM

    def test_free_response_group_is_free_response(self):
        adapter = _StubAdapter(free_response={"g1"}, require_mention=True)
        assert adapter._reply_policy_for("group", "g1") == ReplyPolicy.FREE_RESPONSE

    def test_mention_gated_group_is_mention_gated(self):
        adapter = _StubAdapter(free_response=set(), require_mention=True)
        assert adapter._reply_policy_for("group", "g2") == ReplyPolicy.MENTION_GATED

    def test_no_mention_requirement_group_is_free_response(self):
        # A group that never requires a mention behaves as free-response.
        adapter = _StubAdapter(free_response=set(), require_mention=False)
        assert adapter._reply_policy_for("group", "g3") == ReplyPolicy.FREE_RESPONSE

    def test_build_source_stamps_reply_policy(self):
        """The stamp lands on the SessionSource the turn actually runs on."""
        adapter = _StubAdapter(free_response={"g1"}, require_mention=True)
        source = adapter.build_source(chat_id="g1", chat_type="group")
        assert source.reply_policy == ReplyPolicy.FREE_RESPONSE

    def test_resolve_reply_policy_matches_stamping(self):
        # Sanity: the pure resolver and the adapter hook agree.
        assert resolve_reply_policy(
            is_group=True, in_free_response_set=True, require_mention=True
        ) == ReplyPolicy.FREE_RESPONSE


class TestGateArmsOnlyForFreeResponseTool:
    def test_arms_for_free_response_group_in_tool_mode(self):
        assert _gate_arms(ReplyPolicy.FREE_RESPONSE, mode="tool") is True

    def test_does_not_arm_for_mention_gated(self):
        assert _gate_arms(ReplyPolicy.MENTION_GATED, mode="tool") is False

    def test_does_not_arm_for_dm(self):
        assert _gate_arms(ReplyPolicy.DM, mode="tool") is False

    def test_does_not_arm_in_prompt_mode(self):
        # Shipped default mode never arms the tool gate, even for free-response.
        assert _gate_arms(ReplyPolicy.FREE_RESPONSE, mode="prompt") is False


# --------------------------------------------------------------------------- #
# deterministic privacy gate on the real final-delivery seam
# --------------------------------------------------------------------------- #
async def _deliver(adapter, response, agent_result=None):
    """Run ``response`` through the REAL ``_hmwa_deliver_turn_response`` seam.

    Returns the text the caller would hand to the adapter (``""``/None ==
    suppressed, nothing reaches the wire). Every lane — direct completion,
    queued follow-up, streamed turn — converges on this seam for its final send.
    """
    runner = _make_runner(adapter)
    source = SessionSource(
        platform=adapter.platform, chat_id="-1001", chat_type="group",
        thread_id="17585", reply_policy=ReplyPolicy.FREE_RESPONSE,
    )
    event = MessageEvent(
        text="hi", message_type=MessageType.TEXT, source=source, message_id="1"
    )

    class _Entry:
        session_id = "s1"

    return await runner._hmwa_deliver_turn_response(
        event, source, _Entry(), _SESSION_KEY, None,
        agent_result or {"final_response": response, "messages": [], "api_calls": 1},
        [], response, None, False,
    )


class TestPrivacyGateOnDeliverySeam:
    @pytest.mark.asyncio
    async def test_clean_response_passes_through(self):
        adapter = ProgressCaptureAdapter()
        out = await _deliver(adapter, _CLEAN)
        # Clean text is returned for the caller to send (not suppressed).
        assert out == _CLEAN

    @pytest.mark.asyncio
    async def test_leak_is_suppressed_on_direct_lane(self):
        adapter = ProgressCaptureAdapter()
        out = await _deliver(adapter, _LEAK)
        assert (out or "") == ""
        assert adapter.sent == []

    @pytest.mark.asyncio
    async def test_leak_is_suppressed_on_queued_lane(self):
        # Queued follow-up hands the first turn's result back to this same seam.
        adapter = ProgressCaptureAdapter()
        result = {"final_response": _LEAK, "messages": [], "api_calls": 2,
                  "queued_followup": True}
        out = await _deliver(adapter, _LEAK, agent_result=result)
        assert (out or "") == ""
        assert adapter.sent == []

    @pytest.mark.asyncio
    async def test_leak_is_suppressed_on_streamed_lane(self):
        # A streamed turn's tail still returns through this seam for its final send.
        adapter = ProgressCaptureAdapter()
        result = {"final_response": _LEAK, "messages": [], "api_calls": 1,
                  "streamed": True}
        out = await _deliver(adapter, _LEAK, agent_result=result)
        assert (out or "") == ""
        assert adapter.sent == []
