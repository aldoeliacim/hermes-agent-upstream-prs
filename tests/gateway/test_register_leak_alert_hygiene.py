"""Held-reply owner alert must not re-leak what it held (incident 2026-09-26).

The register guard held a contact reply carrying a compaction summary with a
Radarr API key, then pasted the raw body into the owner-home alert. It also
labelled the WhatsApp pushName ("Hey!") as if it were a verified person.
"""

from types import SimpleNamespace

import pytest

from gateway.run_notifications import GatewayNotificationsMixin

_KEY = "6ed917cdee324555aab7d2717a70a015"


class _Harness(GatewayNotificationsMixin):
    def __init__(self):
        self.sent = []

    async def _notify_owner_home(self, source, note, *, context, media=()):
        self.sent.append((note, context))
        return True


def _source(**kw):
    base = dict(
        user_name="Hey!",
        user_id="5215500000000@s.whatsapp.net",
        chat_id="100000000000001@lid",
        chat_type="dm",
        chat_name="Hey!",
    )
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.mark.asyncio
@pytest.mark.parametrize("salvaged", [False, True])
async def test_alert_body_is_redacted(salvaged):
    h = _Harness()
    body = f"¡Entendido!\n<summary>\n- *Radarr API Key:* `{_KEY}`\n</summary>"
    await h._reroute_leaked_reply_to_owner(_source(), body, salvaged=salvaged)
    note, _ = h.sent[0]
    assert _KEY not in note
    assert "¡Entendido!" in note  # body kept so the owner can judge FP / resend


@pytest.mark.asyncio
async def test_alert_marks_display_name_unverified_and_names_chat_id():
    h = _Harness()
    await h._reroute_leaked_reply_to_owner(_source(), "hola")
    note, _ = h.sent[0]
    assert "unverified" in note.lower()
    assert "100000000000001@lid" in note
    assert "meant for *Hey!*" not in note


@pytest.mark.asyncio
async def test_group_alert_keeps_group_surface():
    h = _Harness()
    src = _source(chat_type="group", chat_name="Familia", chat_id="1203@g.us")
    await h._reroute_leaked_reply_to_owner(src, "hola")
    note, _ = h.sent[0]
    assert "group" in note and "Familia" in note and "1203@g.us" in note


# --- MEDIA: directives in a held body (Kevin DM hold, 2026-09-27) ------------
# The held reply ended with ``MEDIA:/…/FICHAS.md``. The alert pasted it as
# literal text: the owner saw a raw local path and never got the file.


class _MediaHarness(GatewayNotificationsMixin):
    def __init__(self):
        self.sent = []

    async def _notify_owner_home(self, source, note, *, context, media=()):
        self.sent.append((note, context, list(media)))
        return True


@pytest.mark.asyncio
async def test_held_media_directive_becomes_attachment_not_raw_text(tmp_path):
    doc = tmp_path / "FICHAS.md"
    doc.write_text("x")
    h = _MediaHarness()
    await h._reroute_leaked_reply_to_owner(
        _source(), f"Terminé la auditoría.\n\nMEDIA:{doc}\n\nVeredicto."
    )
    (note, _ctx, media), = h.sent
    assert "MEDIA:" not in note
    assert str(doc) not in note
    assert "FICHAS.md" in note
    assert media == [str(doc)]


@pytest.mark.asyncio
async def test_notify_owner_home_sends_held_media_as_document(monkeypatch, tmp_path):
    doc = tmp_path / "FICHAS.md"
    doc.write_text("x")
    calls = []

    class _Adapter:
        async def send(self, chat_id, text):
            calls.append(("send", chat_id, text))

        async def send_document(self, chat_id, file_path, caption=None, file_name=None):
            calls.append(("doc", chat_id, file_path))

    import gateway.delivery as delivery

    monkeypatch.setattr(
        delivery,
        "resolve_delivery_transport",
        lambda plat, config, adapters: SimpleNamespace(adapter=_Adapter()),
    )

    class _Real(GatewayNotificationsMixin):
        pass

    g = _Real()
    g.config = SimpleNamespace(
        platforms={"whatsapp": SimpleNamespace(home_channel=SimpleNamespace(chat_id="HOME"))}
    )
    g.adapters = {}
    ok = await g._notify_owner_home(
        SimpleNamespace(platform="whatsapp"),
        "note",
        context="register_leak_reroute",
        media=[str(doc), str(tmp_path / "missing.md")],
    )
    assert ok
    assert calls == [("send", "HOME", "note"), ("doc", "HOME", str(doc))]
