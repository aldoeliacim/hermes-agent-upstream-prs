"""Non-owner USER.md identity gate — gateway side.

Ported from the live ref (828ce02edf:tests/gateway/test_user_profile_identity.py)
and adapted to the 9.21 candidate. Fixtures use neutral placeholders only.

Covers:
  * _should_skip_user_profile_for_source: skip for a contact, keep for the
    owner (config or user_config home), keep for LOCAL, and the WhatsApp
    LID<->phone-JID owner-in-a-group recognition (and non-owner rejection).
  * build_session_context_prompt: names the Current Session Context as the
    authoritative identity (positive half of the gate).
  * cache isolation: _agent_config_signature MUST differ across the
    owner/non-owner boundary so a cached AIAgent is never reused across it.
"""
from types import SimpleNamespace

from gateway.config import HomeChannel, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_context_prompt, SessionContext


def _runner(home_channel=None):
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(
        get_home_channel=lambda platform: (
            home_channel if home_channel and platform == home_channel.platform else None
        )
    )
    return runner


def test_gateway_skips_global_user_profile_for_non_owner_source():
    runner = _runner(
        HomeChannel(platform=Platform.TELEGRAM, chat_id="owner-chat", name="Home")
    )
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="contact-chat",
        chat_type="dm",
        user_id="contact-user",
        user_name="Contact",
    )

    assert runner._should_skip_user_profile_for_source(
        source=source,
        session_key="agent:main:telegram:dm:contact-chat",
        user_config={},
    ) is True


def test_gateway_keeps_global_user_profile_for_home_channel_source():
    runner = _runner(
        HomeChannel(platform=Platform.TELEGRAM, chat_id="owner-chat", name="Home")
    )
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="owner-chat",
        chat_type="dm",
        user_id="owner-user",
        user_name="Owner",
    )

    assert runner._should_skip_user_profile_for_source(
        source=source,
        session_key="agent:main:telegram:dm:owner-chat",
        user_config={},
    ) is False


def test_gateway_keeps_global_user_profile_for_config_home_channel_source():
    runner = _runner()
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="owner-channel",
        chat_type="dm",
        user_id="owner-user",
        user_name="Owner",
    )

    assert runner._should_skip_user_profile_for_source(
        source=source,
        session_key="agent:main:discord:dm:owner-channel",
        user_config={
            "platforms": {
                "discord": {
                    "home_channel": {
                        "chat_id": "owner-channel",
                    }
                }
            }
        },
    ) is False


def test_gateway_keeps_global_user_profile_for_local_source():
    runner = _runner()
    source = SessionSource(
        platform=Platform.LOCAL,
        chat_id="local",
        chat_type="dm",
    )

    assert runner._should_skip_user_profile_for_source(
        source=source,
        session_key="agent:main:local:dm",
        user_config={},
    ) is False


def test_gateway_fail_closed_skips_when_no_home_configured():
    """FAIL-CLOSED: with NO home channel configured, ownership cannot be proven,
    so a non-LOCAL source has USER.md hidden (inconvenience) rather than leaked
    (incident)."""
    runner = _runner()  # no home channel at all
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="some-chat",
        chat_type="dm",
        user_id="some-user",
        user_name="Someone",
    )

    assert runner._should_skip_user_profile_for_source(
        source=source,
        session_key="agent:main:telegram:dm:some-chat",
        user_config={},
    ) is True


def _write_lid_mapping(tmp_home, lid_digits, phone_jid):
    """Create the bridge lid-mapping file the canonicalizer reads."""
    mapping_dir = tmp_home / "whatsapp" / "session"
    mapping_dir.mkdir(parents=True, exist_ok=True)
    import json
    (mapping_dir / f"lid-mapping-{lid_digits}.json").write_text(
        json.dumps(phone_jid), encoding="utf-8"
    )


def test_gateway_keeps_user_profile_for_owner_lid_in_whatsapp_group(tmp_path, monkeypatch):
    """The owner speaking from inside a WhatsApp GROUP arrives as an opaque LID,
    not a phone number. The gate must expand the LID back to the phone-JID home
    channel and recognize the owner — otherwise USER.md is hidden and the agent
    treats the owner as a stranger in their own group."""
    tmp_home = tmp_path / "hermes-home"
    tmp_home.mkdir(parents=True, exist_ok=True)
    _write_lid_mapping(tmp_home, "55510000000001", "5215551000001@s.whatsapp.net")
    monkeypatch.setenv("HERMES_HOME", str(tmp_home))

    runner = _runner(
        HomeChannel(
            platform=Platform.WHATSAPP,
            chat_id="5215551000001@s.whatsapp.net",
            name="Home",
        )
    )
    # Owner posting in a group: user_id is the opaque LID, chat_id is the group.
    source = SessionSource(
        platform=Platform.WHATSAPP,
        chat_id="5215551000001-1503324008@g.us",
        chat_type="group",
        user_id="55510000000001@lid",
        user_name="Owner",
    )

    assert runner._should_skip_user_profile_for_source(
        source=source,
        session_key="agent:main:whatsapp:group:5215551000001-1503324008@g.us",
        user_config={},
    ) is False


def test_gateway_skips_user_profile_for_non_owner_lid_in_whatsapp_group(tmp_path, monkeypatch):
    """A different group member (non-owner) whose LID maps to their OWN phone
    must still have USER.md hidden — expansion must not leak owner status to
    other members, even though the group JID shares the owner's phone prefix."""
    tmp_home = tmp_path / "hermes-home"
    tmp_home.mkdir(parents=True, exist_ok=True)
    _write_lid_mapping(tmp_home, "55510000000001", "5215551000001@s.whatsapp.net")
    _write_lid_mapping(tmp_home, "55510000000002", "5215551000003@s.whatsapp.net")
    monkeypatch.setenv("HERMES_HOME", str(tmp_home))

    runner = _runner(
        HomeChannel(
            platform=Platform.WHATSAPP,
            chat_id="5215551000001@s.whatsapp.net",
            name="Home",
        )
    )
    # A non-owner member posting in the owner's group.
    source = SessionSource(
        platform=Platform.WHATSAPP,
        chat_id="5215551000001-1503324008@g.us",
        chat_type="group",
        user_id="55510000000002@lid",
        user_name="Member",
    )

    assert runner._should_skip_user_profile_for_source(
        source=source,
        session_key="agent:main:whatsapp:group:5215551000001-1503324008@g.us",
        user_config={},
    ) is True


def test_owner_gate_partial_mapping_only_reverse_file(tmp_path, monkeypatch):
    """Even when only the *reverse* lid-mapping file exists (the bridge wrote one
    direction), expanding the full alias set on both sides must still match the
    owner — a partial mapping shouldn't lock the owner out."""
    tmp_home = tmp_path / "hermes-home"
    (tmp_home / "whatsapp" / "session").mkdir(parents=True, exist_ok=True)
    import json
    (tmp_home / "whatsapp" / "session" / "lid-mapping-55510000000001_reverse.json").write_text(
        json.dumps("5215551000001@s.whatsapp.net"), encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_home))

    runner = _runner(
        HomeChannel(
            platform=Platform.WHATSAPP,
            chat_id="5215551000001@s.whatsapp.net",
            name="Home",
        )
    )
    source = SessionSource(
        platform=Platform.WHATSAPP,
        chat_id="5215551000001-1503324008@g.us",
        chat_type="group",
        user_id="55510000000001@lid",
        user_name="Owner",
    )

    assert runner._should_skip_user_profile_for_source(
        source=source,
        session_key="agent:main:whatsapp:group:5215551000001-1503324008@g.us",
        user_config={},
    ) is False


def test_session_context_marks_current_source_as_authoritative_identity():
    context = SessionContext(
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="contact-chat",
            chat_type="dm",
            user_id="contact-user",
            user_name="Contact",
        ),
        connected_platforms=[Platform.TELEGRAM],
        home_channels={},
    )

    prompt = build_session_context_prompt(context)

    assert "Current Session Context above is authoritative" in prompt
    assert "Do not address the current speaker as the owner" in prompt


def _sig(runner, *, skip_user_profile):
    return runner._agent_config_signature(
        "model-x",
        {"api_key": "k", "base_url": "b", "provider": "p"},
        ["memory"],
        "ephemeral",
        skip_context_files=False,
        skip_user_profile=skip_user_profile,
    )


def test_cache_signature_differs_across_owner_boundary():
    """THE LEAK CLASS: an owner turn and a contact turn under one session key
    must NOT collide in the agent cache. The signature must change with
    skip_user_profile, so the cached AIAgent (whose frozen prompt bakes the
    owner profile in or out) is never reused across the boundary."""
    runner = _runner()
    owner_sig = _sig(runner, skip_user_profile=False)
    contact_sig = _sig(runner, skip_user_profile=True)
    assert owner_sig != contact_sig
    # Determinism: same flag → same signature (cache hit within one silo).
    assert owner_sig == _sig(runner, skip_user_profile=False)
    assert contact_sig == _sig(runner, skip_user_profile=True)


def test_every_gateway_agent_construction_decides_the_owner_profile():
    """Class gate: no gateway code path may build an AIAgent that silently inherits USER.md.

    Every ``AIAgent(...)`` constructed under ``gateway/`` must either pass
    ``skip_user_profile=`` (derived from ``_should_skip_user_profile_for_source``) or build
    a memory-less agent (``skip_memory=``: the hygiene/compression helpers, which load no
    USER.md). A new construction site that forgets both fails here instead of shipping the
    owner's personal profile into a contact's chat.
    """
    import ast
    import pathlib

    decided = {"skip_user_profile", "skip_memory"}

    def _spread_decides(func_node: ast.AST, name: str) -> bool:
        """A ``**name`` spread counts only if the same function assigns ``name["skip_user_profile"]``
        or builds ``name`` with that key. A bare ``**kwargs`` of unknown content does not."""
        for sub in ast.walk(func_node):
            if isinstance(sub, ast.Assign):
                for tgt in sub.targets:
                    if (isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name)
                            and tgt.value.id == name and isinstance(tgt.slice, ast.Constant)
                            and tgt.slice.value in decided):
                        return True
                    if (isinstance(tgt, ast.Name) and tgt.id == name and isinstance(sub.value, ast.Dict)
                            and any(isinstance(k, ast.Constant) and k.value in decided
                                    for k in sub.value.keys)):
                        return True
        return False

    root = pathlib.Path(__file__).resolve().parents[2] / "gateway"
    offenders = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for func_node in ast.walk(tree):
            if not isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(func_node):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
                if name != "AIAgent":
                    continue
                if {kw.arg for kw in node.keywords if kw.arg} & decided:
                    continue
                spreads = [kw.value.id for kw in node.keywords
                           if kw.arg is None and isinstance(kw.value, ast.Name)]
                if any(_spread_decides(func_node, s) for s in spreads):
                    continue
                offenders.append(f"{path.relative_to(root.parent)}:{node.lineno}")
    assert offenders == [], f"AIAgent built without an owner-profile decision: {sorted(set(offenders))}"
