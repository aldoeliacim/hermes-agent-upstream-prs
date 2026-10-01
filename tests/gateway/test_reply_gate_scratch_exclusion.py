"""Reply-gate scratch exclusion — breaking the verdict-imitation feedback loop.

The bug class (measured live 2026-08-11, group session 20260811_023058): a
tool-gated turn's suppressed free-text tail — almost always a stay-silent
verdict like "No respondo — plática entre ellos" — is persisted as a normal
assistant row. The next turn REPLAYS that row to the model as something it
previously said, and the model imitates the shape: one replayed verdict seeded
20+ near-identical verdict rows in a single session. Those verdicts are
exactly the leaked-narration shapes gateway/response_filters.py exists to
suppress — i.e. the system was training the model to produce the leak the
filters then have to catch.

The mechanism fix has two halves, each pinned here:
  1. gateway/run.py's reply-gate suppression block tags the suppressed tail
     ``display_kind='reply_gate_scratch'`` (via the existing
     set_latest_matching_message_display_kind writer).
  2. _build_gateway_agent_history EXCLUDES tagged rows from the API replay —
     the model never sees its own suppressed verdicts again. Transcripts and
     exports keep the full record; only the model's view drops them.
"""

from gateway.run import _build_gateway_agent_history


def _mk_history():
    return [
        {"role": "user", "content": "[Riley] Hola", "timestamp": 1000.0},
        {
            "role": "assistant",
            "content": "No respondo — es un saludo al grupo sin dirigirse a mí.",
            "display_kind": "reply_gate_scratch",
            "timestamp": 1001.0,
        },
        {"role": "user", "content": "[Alex] Amooos", "timestamp": 1002.0},
        {
            "role": "assistant",
            "content": "No respondo — chat casual del grupo.",
            "display_kind": "reply_gate_scratch",
            "timestamp": 1003.0,
        },
        {"role": "user", "content": "[Riley] @HAL ¿tú qué opinas?", "timestamp": 1004.0},
        {
            "role": "assistant",
            "content": "Yo digo que el jueves les rinde mejor.",
            "timestamp": 1005.0,
        },
    ]


class TestReplyGateScratchExclusion:
    def test_scratch_rows_are_excluded_from_replay(self):
        agent_history, _ = _build_gateway_agent_history(_mk_history())
        contents = [m.get("content", "") for m in agent_history]
        # The delivered answer stays; both suppressed verdicts are gone.
        assert any("jueves les rinde" in c for c in contents)
        assert not any("No respondo" in c for c in contents)

    def test_scratch_exclusion_only_applies_to_assistant_rows(self):
        # A user row carrying the same display_kind (defensive: should never
        # happen, but a tagging bug must not eat user messages) is preserved.
        history = [
            {
                "role": "user",
                "content": "[Robin] hola",
                "display_kind": "reply_gate_scratch",
                "timestamp": 1000.0,
            },
            {"role": "assistant", "content": "Hola Robin", "timestamp": 1001.0},
        ]
        agent_history, _ = _build_gateway_agent_history(history)
        contents = [m.get("content", "") for m in agent_history]
        assert any("hola" in c for c in contents)

    def test_other_display_kinds_still_replay(self):
        # display_kind is a general presentation column (synthetic inputs,
        # reactions metadata...). Only the reply_gate_scratch value is
        # replay-excluded.
        history = [
            {"role": "user", "content": "hola", "timestamp": 1000.0},
            {
                "role": "assistant",
                "content": "respuesta normal",
                "display_kind": "synthetic_input",
                "timestamp": 1001.0,
            },
        ]
        agent_history, _ = _build_gateway_agent_history(history)
        contents = [m.get("content", "") for m in agent_history]
        assert any("respuesta normal" in c for c in contents)
