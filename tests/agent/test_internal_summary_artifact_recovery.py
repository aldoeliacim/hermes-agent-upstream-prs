"""Regression: internal handoff summaries must never become user replies."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


INTERNAL_ARTIFACT = """<summary>
## Overview
The previous task was still in progress.
## Next Steps
Wait for the background operation to finish.
</summary>"""


@pytest.fixture()
def loop_agent():
    from run_agent import AIAgent

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="http://router.invalid/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.tool_delay = 0
        agent.compression_enabled = False
        agent.save_trajectories = False
        return agent


def _response(text: str):
    from tests.agent.test_run_agent import _mock_response

    return _mock_response(content=text, finish_reason="stop")


def test_internal_summary_is_retried_and_never_persisted(loop_agent):
    loop_agent.client.chat.completions.create.side_effect = [
        _response(INTERNAL_ARTIFACT),
        _response("Here is the answer to your current question."),
    ]

    result = loop_agent.run_conversation("Are you available?")

    assert result["final_response"] == "Here is the answer to your current question."
    assert loop_agent.client.chat.completions.create.call_count == 2
    assert all(
        "<summary>" not in str(message.get("content") or "").lower()
        for message in result["messages"]
        if isinstance(message, dict)
    )


def test_repeated_internal_summaries_fail_safe(loop_agent):
    loop_agent.client.chat.completions.create.side_effect = [
        _response(INTERNAL_ARTIFACT),
        _response(INTERNAL_ARTIFACT),
        _response(INTERNAL_ARTIFACT),
    ]

    result = loop_agent.run_conversation("Continue with my request.")

    final = str(result["final_response"] or "")
    assert final.strip()
    assert "<summary>" not in final.lower()
    assert all(
        "<summary>" not in str(message.get("content") or "").lower()
        for message in result["messages"]
        if isinstance(message, dict)
    )


def test_iteration_budget_fallback_cannot_reintroduce_summary(loop_agent):
    loop_agent.max_iterations = 1
    loop_agent.client.chat.completions.create.side_effect = [
        _response(INTERNAL_ARTIFACT),
        _response(INTERNAL_ARTIFACT),
    ]

    result = loop_agent.run_conversation("Are you available?")

    final = str(result["final_response"] or "")
    assert final.strip()
    assert "<summary>" not in final.lower()
    assert all(
        "<summary>" not in str(message.get("content") or "").lower()
        for message in result["messages"]
        if isinstance(message, dict)
    )


@pytest.mark.parametrize(
    ("text", "user_message"),
    [
        ("The HTML `<summary>` element labels a details widget.", "Explain HTML."),
        ("<summary>Short label</summary>", "Write a concise label."),
        (
            "<summary>\n## Overview\nRequested markup\n## Next Steps\nNone\n</summary>",
            "Return exactly one HTML <summary> element with those headings.",
        ),
        ("Prefix before <summary>internal quote</summary>", "Explain this quote."),
        ("<summary>internal quote</summary> followed by prose", "Explain this quote."),
    ],
)
def test_legitimate_summary_content_is_not_classified(text, user_message):
    from agent.turn_final_response import is_internal_summary_artifact

    assert is_internal_summary_artifact(text, user_message=user_message) is False


def test_fenced_xml_handoff_with_html_headings_is_retried_without_delivery(loop_agent):
    # Distilled from a contact's PR review: the model returned an internal
    # checkpoint instead of its verdict, wrapped in a whole XML code fence.
    artifact = (
        "```xml\n<summary>\n<h2>Overview</h2>\nPR review in progress.\n"
        "<h2>Key Details & Breadcrumbs</h2>\nTests passed.\n"
        "<h2>Current State</h2>\nNext: verify remaining findings.\n"
        "</summary>\n```"
    )
    from agent.turn_final_response import internal_artifact_kind

    assert internal_artifact_kind(artifact, user_message="Review my PR and give a verdict.") == "summary_envelope"
    loop_agent.client.chat.completions.create.side_effect = [
        _response(artifact), _response("Revisión completa: cambios requeridos en T6."),
    ]
    result = loop_agent.run_conversation("Review my PR and give a verdict.")
    assert result["final_response"] == "Revisión completa: cambios requeridos en T6."
    assert loop_agent.client.chat.completions.create.call_count == 2
    assert all(
        "<h2>Current State</h2>" not in str(message.get("content") or "")
        for message in result["messages"] if isinstance(message, dict)
    )


@pytest.mark.parametrize("text", [
    "Here is the requested example:\n```xml\n<summary>\n<h2>Overview</h2>\n"
    "A summary element.\n<h2>Current State</h2>\nA heading.\n</summary>\n```",
    "```xml\n<summary>\n<h2>Overview</h2>\nFacts.\n"
    "<h2>Conclusion</h2>\nDone.\n</summary>\n```",
])
def test_xml_example_or_completed_document_is_not_handoff(text):
    from agent.turn_final_response import internal_artifact_kind

    assert internal_artifact_kind(text) is None


def test_whole_handoff_envelope_is_classified_with_whitespace():
    from agent.turn_final_response import is_internal_summary_artifact

    assert is_internal_summary_artifact(
        "  \n" + INTERNAL_ARTIFACT + "\n  ", user_message="Continue my current task."
    ) is True


def test_prefatory_handoff_is_retried_without_delivery(loop_agent):
    # The incident had a short assistant-facing preface BEFORE the XML block.
    # Keep this synthetic: private details from the original must not enter fixtures.
    artifact = (
        'The user said "OK" and the previous task is complete.\n\n'
        'I need to output the full summary as the previous summary was an internal context handoff.\n'
        '<summary>\n## Overview\nInternal project notes.\n## Current State\nArchived.\n</summary>'
    )
    loop_agent.client.chat.completions.create.side_effect = [
        _response(artifact), _response("Noted."),
    ]
    result = loop_agent.run_conversation("OK")
    assert result["final_response"] == "Noted."
    assert loop_agent.client.chat.completions.create.call_count == 2
    assert all(
        "Internal project notes" not in str(message.get("content") or "")
        for message in result["messages"] if isinstance(message, dict)
    )


def test_prefatory_handoff_without_magic_english_phrase_is_retried(loop_agent):
    artifact = (
        'The task was acknowledged.\n\n'
        '<summary>\n## Overview\nPrivate synthetic note.\n'
        '## Current State\nClosed.\n</summary>\n'
        'The above is for my next context window.'
    )
    loop_agent.client.chat.completions.create.side_effect = [
        _response(artifact), _response('Entendido.'),
    ]
    result = loop_agent.run_conversation('Ok')
    assert result['final_response'] == 'Entendido.'
    assert all('Private synthetic note' not in str(m.get('content') or '')
               for m in result['messages'] if isinstance(m, dict))


def test_prefatory_handoff_does_not_block_requested_xml_summary():
    from agent.turn_final_response import internal_artifact_kind
    artifact = '<summary>\n## Overview\nWork.\n## Current State\nDone.\n</summary>'
    assert internal_artifact_kind(artifact, user_message='Please output an HTML <summary> tag.') is None


def test_summary_recovery_rows_are_ephemeral_to_persistence():
    from agent.session_persistence import (
        _EPHEMERAL_SCAFFOLDING_FLAGS,
        _is_ephemeral_scaffolding,
    )

    assert "_internal_summary_recovery" in _EPHEMERAL_SCAFFOLDING_FLAGS
    assert _is_ephemeral_scaffolding({
        "role": "assistant",
        "content": INTERNAL_ARTIFACT,
        "_internal_summary_recovery": True,
    })
    assert _is_ephemeral_scaffolding({
        "role": "user",
        "content": "retry without summarizing",
        "_internal_summary_recovery": True,
    })


def test_summary_recovery_nudge_is_not_human_intent_during_compression():
    """Preflight compression may run before the retry pair is popped."""
    from agent.conversation_compression import _is_real_user_message

    assert not _is_real_user_message({
        "role": "user",
        "content": "retry without summarizing",
        "_internal_summary_recovery": True,
    })



# ---------------------------------------------------------------------------
# Prefix envelope + text-written tool call (live incident shape). A response that
# STARTS with the compaction handoff, continues with a prose line and ends with a
# tool call written as text passed the whole-response match and reached users.
# ---------------------------------------------------------------------------
PREFIX_ARTIFACT = INTERNAL_ARTIFACT + """
I'm still working on getting your updated document. After that, buscaré tu recibo.

<execute_bash>
{"CommandLine":"browser_console 'query_selector download button' 'click'","Cwd":null,"Blocking":true}
</execute_bash>"""

TOOL_MARKUP_ONLY = """Déjame revisar el archivo.

<execute_bash>
{"CommandLine":"ls /tmp/robin","Cwd":null,"Blocking":true}
</execute_bash>"""


@pytest.mark.parametrize(
    "text,kind",
    [
        ("  \n" + INTERNAL_ARTIFACT + "\n  ", "summary_envelope"),
        (PREFIX_ARTIFACT, "summary_envelope"),
        (TOOL_MARKUP_ONLY, "text_tool_markup"),
        ("Voy.\n<function_calls>\n<invoke name=\"terminal\">", "text_tool_markup"),
    ],
)
def test_artifact_shapes_are_classified(text, kind):
    from agent.turn_final_response import internal_artifact_kind

    assert internal_artifact_kind(text) == kind


@pytest.mark.parametrize(
    "text",
    [
        # HTML discussion, inline or as a leading block WITHOUT handoff headings.
        "Use <summary>Details</summary> inside <details> to make it collapsible.",
        "<summary>\nShipping details\n</summary>\nThe package leaves on Monday.",
        # Tool syntax in prose, in a fenced example, and inline JSON.
        "In Devin the shell tool is written as <execute_bash> with a JSON body.",
        "Ejemplo:\n```xml\n<execute_bash>\n{\"CommandLine\": \"ls\"}\n</execute_bash>\n```",
        "Tu pedido: {\"talla\": \"M\"} — listo.",
        "",
    ],
)
def test_legitimate_text_is_not_internal(text):
    from agent.turn_final_response import internal_artifact_kind

    assert internal_artifact_kind(text) is None


def test_explicitly_requested_summary_element_is_not_an_envelope():
    from agent.turn_final_response import internal_artifact_kind

    assert internal_artifact_kind(
        INTERNAL_ARTIFACT, user_message="Return the HTML <summary> element for this section."
    ) is None


@pytest.mark.parametrize("artifact", [PREFIX_ARTIFACT, TOOL_MARKUP_ONLY])
def test_prefix_and_markup_artifacts_are_retried_not_delivered(loop_agent, artifact):
    """The prefix and text-tool shapes take the same recovery path as a whole envelope."""
    loop_agent.client.chat.completions.create.side_effect = [
        _response(artifact),
        _response("Sigo con tu documento, Robin. En cuanto lo tenga te aviso."),
    ]

    result = loop_agent.run_conversation("¿Ya quedó?")

    assert result["final_response"] == "Sigo con tu documento, Robin. En cuanto lo tenga te aviso."
    for m in result["messages"]:
        content = str(m.get("content") or "") if isinstance(m, dict) else ""
        assert "<summary>" not in content.lower()
        assert "<execute_bash>" not in content.lower()


def test_text_tool_markup_gets_its_own_nudge(loop_agent):
    """The re-prompt tells the model to use the native tool interface, not to stop summarising."""
    loop_agent.client.chat.completions.create.side_effect = [
        _response(TOOL_MARKUP_ONLY),
        _response("Listo."),
    ]
    loop_agent.run_conversation("Revisa el archivo.")
    second_call = loop_agent.client.chat.completions.create.call_args_list[1]
    sent = second_call.kwargs.get("messages") or second_call.args[0]
    nudges = [m["content"] for m in sent if m.get("role") == "user" and "native" in str(m.get("content"))]
    assert nudges, "the text_tool_markup nudge must reach the model"
