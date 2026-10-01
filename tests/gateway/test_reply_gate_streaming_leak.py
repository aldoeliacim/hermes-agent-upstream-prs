"""Reply gate vs. streaming: a decided turn must never leak a trailing bubble.

CONFIRMED PRODUCTION BUG (2026-08-10, three separate turns, Telegram DM
owner chat). The model called ``send_message(target="current", ...)`` — the real
answer, delivered — then wrote a short trailing narration ("Entregado.").  The
post-turn reply gate DID suppress that tail::

    02:13:45,469 Suppressing normal final send ... (streamed=True content_delivered=True)
    02:13:45,487 reply_gate: tool-mode chars=10 tool_sends=1 suppressed=True outcome=replied

…but 18ms too late: the stream consumer had already pushed the tail to the
platform token-by-token.  ``suppressed=True`` only kills the *final send*; it
cannot recall tokens already on screen.  The user saw two bubbles.

Why it hid for so long: ``display.platforms.whatsapp.streaming`` is false, so
WhatsApp masked the bug entirely.  It reproduces on any surface with streaming
enabled — the leak is in the gate/stream interaction, not in one adapter.

The structural guarantee under test: the streaming path and the post-turn
delivery block consult the SAME predicate
(``TurnDeliveryState.delivery_decided``).  Streaming is a real delivery path,
so once the turn has decided, streaming stops emitting — no second bubble is
reachable, on any platform, regardless of timing.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from tools.approval import (
    TurnDeliveryState,
    set_turn_delivery_state,
    reset_turn_delivery_state,
    note_current_message_delivered,
    note_current_message_silent,
)


def _make_adapter():
    """Adapter that records every outbound bubble and deletion."""
    adapter = MagicMock()
    adapter.REQUIRES_EDIT_FINALIZE = False
    adapter.MAX_MESSAGE_LENGTH = 4096
    adapter.send = AsyncMock(
        return_value=MagicMock(success=True, message_id="m1")
    )
    adapter.edit_message = AsyncMock(
        return_value=MagicMock(success=True, message_id="m1")
    )
    adapter.delete_message = AsyncMock(return_value=True)
    return adapter


def _make_consumer(adapter, decided_flag):
    return GatewayStreamConsumer(
        adapter=adapter,
        chat_id="chat-1",
        config=StreamConsumerConfig(edit_interval=0.0, buffer_threshold=1),
        delivery_decided=lambda: decided_flag(),
    )


# ── TurnDeliveryState: the canonical predicate ───────────────────────────


class TestDeliveryDecidedPredicate:
    def test_fresh_turn_is_undecided(self):
        assert TurnDeliveryState().delivery_decided is False

    def test_a_current_chat_send_decides_the_turn(self):
        state = TurnDeliveryState()
        state.tool_sends = 1
        assert state.delivery_decided is True

    def test_explicit_silence_also_decides_the_turn(self):
        """Silence is a first-class decision, not the absence of one."""
        state = TurnDeliveryState()
        state.decided_silent = 1
        assert state.delivery_decided is True

    def test_tool_helpers_flip_the_predicate(self):
        """The real tool entry points drive the canonical predicate."""
        state = TurnDeliveryState()
        token = set_turn_delivery_state(state)
        try:
            assert state.delivery_decided is False
            note_current_message_delivered()
            assert state.delivery_decided is True
        finally:
            reset_turn_delivery_state(token)

    def test_silent_helper_flips_the_predicate(self):
        state = TurnDeliveryState()
        token = set_turn_delivery_state(state)
        try:
            note_current_message_silent()
            assert state.delivery_decided is True
        finally:
            reset_turn_delivery_state(token)


# ── The regression itself ────────────────────────────────────────────────


class TestStreamingRespectsTheGate:
    def test_trailing_narration_after_send_is_never_streamed(self):
        """THE BUG: tail written after send_message must not reach the user.

        Without the latch the consumer streams "Entregado." as a second
        bubble; the post-turn gate then reports suppressed=True while the
        user is already looking at the duplicate.
        """
        adapter = _make_adapter()
        decided = {"v": False}
        consumer = _make_consumer(adapter, lambda: decided["v"])

        async def scenario():
            task = asyncio.create_task(consumer.run())
            # The model calls send_message(target="current") — real answer
            # delivered by the tool, gate closes mid-turn.
            decided["v"] = True
            await asyncio.sleep(0.05)
            # Then it writes a trailing narration.
            consumer.on_delta("Entregado.")
            consumer.finish()
            await asyncio.wait_for(task, timeout=5)

        asyncio.run(scenario())

        assert adapter.send.await_count == 0, (
            "Trailing narration leaked to the platform as a second bubble — "
            f"send called {adapter.send.await_count}x with "
            f"{[c.kwargs.get('text') for c in adapter.send.await_args_list]}"
        )

    def test_gate_leaves_delivery_flags_false(self):
        """The gateway must see 'the stream delivered nothing'.

        If the consumer reported content_delivered=True here, the post-turn
        block would trust a delivery that never happened.
        """
        adapter = _make_adapter()
        consumer = _make_consumer(adapter, lambda: True)

        async def scenario():
            task = asyncio.create_task(consumer.run())
            consumer.on_delta("trailing text")
            consumer.finish()
            await asyncio.wait_for(task, timeout=5)

        asyncio.run(scenario())

        assert consumer.final_content_delivered is False
        assert consumer._final_response_sent is False

    def test_partial_bubble_already_on_screen_is_retracted(self):
        """A bubble flushed before the decision must be deleted, not orphaned.

        Timing is not always kind: an interval tick can put a partial tail on
        screen microseconds before send_message lands. Abandoning alone would
        leave that fragment visible forever.
        """
        adapter = _make_adapter()
        decided = {"v": False}
        consumer = _make_consumer(adapter, lambda: decided["v"])

        async def scenario():
            task = asyncio.create_task(consumer.run())
            consumer.on_delta("partial tail that got flushed")
            await asyncio.sleep(0.1)  # let a tick put it on screen
            decided["v"] = True       # decision lands after
            consumer.on_delta(" ...more")
            consumer.finish()
            await asyncio.wait_for(task, timeout=5)

        asyncio.run(scenario())

        if adapter.send.await_count:
            assert adapter.delete_message.await_count >= 1, (
                "A preview bubble was left on screen after the gate closed"
            )
        assert consumer.final_content_delivered is False

    def test_latch_is_permanent(self):
        """Once closed, a later tick may not resume emitting."""
        adapter = _make_adapter()
        flip = {"n": 0}

        def flapping():
            flip["n"] += 1
            return flip["n"] == 1  # decided once, then "undecided" again

        consumer = _make_consumer(adapter, flapping)

        async def scenario():
            task = asyncio.create_task(consumer.run())
            consumer.on_delta("text after the flap")
            consumer.finish()
            await asyncio.wait_for(task, timeout=5)

        asyncio.run(scenario())

        assert consumer._delivery_gate_closed is True
        assert adapter.send.await_count == 0


class TestUndecidedTurnsStreamNormally:
    """The latch must not break ordinary streaming — no over-correction."""

    def test_undecided_turn_streams_its_answer(self):
        adapter = _make_adapter()
        consumer = _make_consumer(adapter, lambda: False)

        async def scenario():
            task = asyncio.create_task(consumer.run())
            consumer.on_delta("A normal streamed answer.")
            consumer.finish()
            await asyncio.wait_for(task, timeout=5)

        asyncio.run(scenario())

        assert adapter.send.await_count >= 1, (
            "Normal streaming regressed — an undecided turn delivered nothing"
        )

    def test_default_callable_keeps_non_gateway_callers_streaming(self):
        """CLI/tests construct the consumer without the latch: unchanged."""
        adapter = _make_adapter()
        consumer = GatewayStreamConsumer(
            adapter=adapter,
            chat_id="chat-1",
            config=StreamConsumerConfig(edit_interval=0.0, buffer_threshold=1),
        )

        async def scenario():
            task = asyncio.create_task(consumer.run())
            consumer.on_delta("plain streaming, no gate involved")
            consumer.finish()
            await asyncio.wait_for(task, timeout=5)

        asyncio.run(scenario())

        assert adapter.send.await_count >= 1
