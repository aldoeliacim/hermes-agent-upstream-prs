"""The batched-completion instruction must not INVITE a status-report leak.

THE INCIDENT (2026-09-07, WhatsApp DM to a contact). Two finished render jobs
were batched into one synthetic turn after the deliverable had already been
sent. The instruction said "absorb it silently", and the model absorbed it the
only way a delivery path allows — by writing the absorption down:

    "The video is already delivered — those were the render jobs behind it.
     Nothing new to report."

That is an owner-report register (narration ABOUT the conversation, addressed
to nobody) delivered into a contact's private chat. Every deterministic guard
passed it: the recipient's name never appears (the third-person fallback is
name-anchored), there is no stay-silent verb ("Nothing new to report" is not
"no respondo" / "I'll stay silent"), and there is no tool jargon. Only the
async semantic auditor flagged it, AFTER delivery.

WHY THIS FILE TESTS THE PROMPT, NOT A PATTERN. Adding "nothing new to report"
to a regex would fix that sentence and no other: "nothing actionable",
"already handled", "same as before" are unbounded paraphrases of one idea.
This repo has twice paid for enumerating an open set (the greeting list in
_is_leaked_silence_narration; the default-PASS rewrite of
mentions_recipient_third_person). The instruction is the SOURCE of the text,
so constraining the source closes the class instead of one phrasing.

The contract asserted here: every batched-completion instruction must name a
CONCRETE emission for "nothing to add" (a silence marker the gateway already
suppresses) and must forbid the narration shape. An instruction that only says
"silently" without naming what to emit is ambiguous on a delivery path — it is
satisfied by describing the silence — and must fail this test.
"""
import re
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from gateway.response_filters import (  # noqa: E402
    LIVE_GATEWAY_SILENT_MARKERS,
    is_intentional_silence_response,
)
from gateway.run import (  # noqa: E402
    _ABSORB_SILENTLY_INSTRUCTION,
    GatewayRunner,
)


def _batched_process_turn(n: int = 2) -> str:
    """Render the real coalesced-completion turn for n finished processes."""
    entries = [
        (
            "",
            {
                "session_id": f"proc_{i}",
                "exit_code": 0,
                "completion_reason": "exited",
                "output": "MoviePy - video ready /tmp/vid/out.mp4\ndone",
            },
            None,
        )
        for i in range(n)
    ]
    return GatewayRunner._format_coalesced_process_completions(entries)


def _batched_delegation_turn(n: int = 2) -> str:
    return GatewayRunner._format_coalesced_async_delegations(
        [f"- subagent {i} finished" for i in range(n)]
    )


ALL_BATCH_TURNS = (_batched_process_turn, _batched_delegation_turn)


def test_instruction_names_a_concrete_silence_emission():
    """"Silently" alone is ambiguous; the turn must say what to EMIT.

    This is the assertion the pre-fix wording ("absorb it silently") fails:
    it never names an emission, so "nothing to add" has no legal action and
    the model invents prose.

    Matched on WORD BOUNDARIES, not substrings. A bare `in` check passes on
    the pre-fix text by accident — "silent**ly**" contains the marker token
    "SILENT" — which would make this a test that cannot fail. Caught by
    running it against the reverted wording; an assertion that survives the
    bug it was written for is not a gate.
    """
    marker_in_instruction = [
        m for m in LIVE_GATEWAY_SILENT_MARKERS
        if re.search(rf"(?<!\w){re.escape(m)}(?!\w)",
                     _ABSORB_SILENTLY_INSTRUCTION, re.IGNORECASE)
    ]
    assert marker_in_instruction, (
        "The instruction must name a silence marker the gateway actually "
        f"suppresses (one of {sorted(LIVE_GATEWAY_SILENT_MARKERS)}) as a "
        f"whole token; got: {_ABSORB_SILENTLY_INSTRUCTION!r}"
    )
    # And that named marker must really be suppressed by the delivery path,
    # so the instruction can never point at a token that would be delivered.
    for marker in marker_in_instruction:
        assert is_intentional_silence_response(marker), (
            f"instruction names {marker!r} but the gateway would DELIVER it"
        )


def test_instruction_forbids_the_status_report_shape():
    """The exact leaked shapes must be named as prohibited, not merely unasked.

    Keyed on the IDEA ("nothing new", "already delivered", "no reply needed"),
    which is what the model reached for, rather than on the incident's literal
    sentence.
    """
    low = _ABSORB_SILENTLY_INSTRUCTION.lower()
    assert re.search(r"\bdo not\b|\bdon't\b|\bnever\b", low), (
        "instruction must contain an explicit prohibition"
    )
    for idea in ("nothing new", "already delivered", "no reply"):
        assert idea in low, f"instruction does not forbid the {idea!r} shape"


def test_every_batched_turn_carries_the_instruction():
    """Both fan-in formatters must share one instruction — no drift.

    The process batch and the delegation batch are separate formatters that
    historically embedded the same sentence twice. Two copies is how one gets
    fixed and the other keeps leaking.
    """
    for build in ALL_BATCH_TURNS:
        turn = build()
        assert _ABSORB_SILENTLY_INSTRUCTION in turn, (
            f"{build.__name__} does not carry the shared instruction"
        )


def test_no_batched_turn_asks_for_bare_silent_absorption():
    """THE REGRESSION GATE — fails on the exact pre-fix wording.

    "absorb it silently" tells the model to be silent without telling it what
    to emit. On a path where every turn produces a message, that is an
    instruction to WRITE the absorption. No batch turn may ask for it again.
    """
    for build in ALL_BATCH_TURNS:
        turn = build().lower()
        assert "absorb it silently" not in turn, (
            f"{build.__name__} still asks the model to 'absorb it silently' — "
            "the wording that produced the 2026-09-07 DM leak"
        )


def test_leaked_sentence_is_not_something_the_turn_asks_for():
    """The incident text must not be a compliant reading of the instruction.

    Belt-and-braces on the two ideas the leak was built from: if the turn
    still merely says "silently" and never forbids a status line, the leaked
    sentence remains a valid way to obey it.
    """
    leaked = "The video is already delivered — those were the render jobs behind it. Nothing new to report."
    # The guard stack genuinely does not catch it; that is WHY the prompt is
    # the fix. Asserting this keeps the test honest about what protects us.
    assert not is_intentional_silence_response(leaked), (
        "if a deterministic guard now catches this, revisit whether the "
        "prompt fix is still the primary defence"
    )
    for build in ALL_BATCH_TURNS:
        turn = build().lower()
        assert "nothing new" in turn and (
            "do not" in turn or "don't" in turn or "never" in turn
        ), f"{build.__name__} does not rule the leaked shape out"
