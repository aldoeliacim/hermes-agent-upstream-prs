"""Cross-identity delivery guard: a non-owner's cron job must not report to the owner.

Bug (2026-09-11, job 6f43f6241fbe): a contact asked in his own WhatsApp DM
for a price watch; the agent created the job with ``deliver`` set to the
OWNER's home channel. ``origin`` correctly recorded the contact, so the
provenance was there — nothing compared it against the target.

The invariant is narrow on purpose (see cron/delivery_guard.py):

    BLOCK  iff  requester is a known NON-owner  AND  a deliver element
                resolves to an owner home channel.

It is NOT "origin must equal deliver": a contact may legitimately ask for
delivery into a group they share with the owner (the family-group plant
reminder is correct and must stay creatable).

Structure:
  * a MATRIX over (requester kind) x (deliver form) asserting the verdict per
    cell, with failure messages naming the cell;
  * E2E through the real ``cronjob()`` create and update paths against a temp
    cron store, with the session env the gateway sets for a chat turn.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from gateway.session_context import clear_session_vars, set_session_vars

# --- fixed identities for the matrix -------------------------------------
# Owner home channels as the scheduler reports them (env mirror / config).
OWNER_TG = "900000001"
OWNER_WA_PHONE = "5215550000001@s.whatsapp.net"
OWNER_WA_LID = "90000000000001@lid"          # aliases OWNER_WA_PHONE
CONTACT_WA_LID = "80000000000002@lid"        # a non-owner DM
CONTACT_TG = "111222333"
SHARED_GROUP = "5215550000001-1000000001@g.us"

_HOME = {"telegram": OWNER_TG, "whatsapp": OWNER_WA_PHONE}
_WA_ALIASES = {
    "5215550000001": {"5215550000001", "90000000000001"},
    "90000000000001": {"5215550000001", "90000000000001"},
}


def _fake_home(platform: str) -> str:
    return _HOME.get(platform.lower(), "")


def _fake_expand(identifier: str):
    base = identifier.split("@", 1)[0].lstrip("+")
    return set(_WA_ALIASES.get(base, {base}))


@pytest.fixture(autouse=True)
def _pinned_identities():
    """Pin home channels + WA alias map so the matrix is hermetic."""
    with patch("cron.scheduler_delivery._get_home_target_chat_id", side_effect=_fake_home), patch(
        "gateway.whatsapp_identity.expand_whatsapp_aliases", side_effect=_fake_expand
    ):
        yield


# --- the matrix -----------------------------------------------------------
# requester kind -> origin dict (None = unknown / CLI / cron-context)
REQUESTERS = {
    "owner-tg": {"platform": "telegram", "chat_id": OWNER_TG, "chat_name": "Owner"},
    "owner-wa-phone": {"platform": "whatsapp", "chat_id": OWNER_WA_PHONE, "chat_name": "Owner"},
    "owner-wa-lid": {"platform": "whatsapp", "chat_id": OWNER_WA_LID, "chat_name": "Owner"},
    "contact-wa": {"platform": "whatsapp", "chat_id": CONTACT_WA_LID, "chat_name": "Quetzalcoatl"},
    "contact-tg": {"platform": "telegram", "chat_id": CONTACT_TG, "chat_name": "Someone"},
    "unknown": None,
}

# deliver form -> does it address the owner?
DELIVERS = {
    "omitted": (None, False),
    "origin": ("origin", False),
    "local": ("local", False),
    "bare-telegram": ("telegram", True),
    "bare-whatsapp": ("whatsapp", True),
    "all": ("all", True),
    "owner-tg-explicit": (f"telegram:{OWNER_TG}", True),
    "owner-tg-thread": (f"telegram:{OWNER_TG}:47593", True),
    "owner-wa-phone-explicit": (f"whatsapp:{OWNER_WA_PHONE}", True),
    "owner-wa-lid-explicit": (f"whatsapp:{OWNER_WA_LID}", True),      # alias of home
    "contact-wa-explicit": (f"whatsapp:{CONTACT_WA_LID}", False),
    "shared-group": (f"whatsapp:{SHARED_GROUP}", False),                # not the home DM
    "bot-chat": ("bot-chat", False),
    "combo-origin+local": ("origin,local", False),
    "combo-origin+owner": (f"origin,telegram:{OWNER_TG}", True),        # one bad element taints
    "combo-contact+bare-tg": (f"whatsapp:{CONTACT_WA_LID},telegram", True),
}


def _expected(requester_kind: str, targets_owner: bool) -> bool:
    """True = blocked."""
    if requester_kind == "unknown" or requester_kind.startswith("owner"):
        return False
    return targets_owner


@pytest.mark.parametrize("requester_kind", list(REQUESTERS))
@pytest.mark.parametrize("deliver_kind", list(DELIVERS))
def test_matrix(requester_kind, deliver_kind):
    from cron.delivery_guard import check_cross_identity_delivery

    origin = REQUESTERS[requester_kind]
    deliver, targets_owner = DELIVERS[deliver_kind]
    err = check_cross_identity_delivery(origin, deliver)
    blocked = err is not None
    want = _expected(requester_kind, targets_owner)
    assert blocked == want, (
        f"cell requester={requester_kind} deliver={deliver_kind} ({deliver!r}): "
        f"expected {'BLOCK' if want else 'ALLOW'}, got {'BLOCK' if blocked else 'ALLOW'}"
        + (f" — {err}" if err else "")
    )


def test_blocked_message_names_fix():
    from cron.delivery_guard import check_cross_identity_delivery

    err = check_cross_identity_delivery(REQUESTERS["contact-wa"], "whatsapp")
    assert err
    for needle in ("non-owner", "Quetzalcoatl", "origin", "whatsapp:<their chat id>"):
        assert needle in err, f"message must name {needle!r}: {err}"


def test_requester_on_platform_without_home_is_not_owner():
    """No home configured on the requester's platform: cannot prove owner → non-owner."""
    from cron.delivery_guard import check_cross_identity_delivery

    origin = {"platform": "discord", "chat_id": "42", "chat_name": "x"}
    assert check_cross_identity_delivery(origin, "telegram") is not None


# Home resolution can come back EMPTY (env mirror not loaded in this process,
# config.yaml without a home block, a platform added after the guard). A
# guard that reads "no home known" as "target is safe" fails open — the exact
# class it exists to close. Every deliver form that addresses a home must
# still block for a non-owner when no home id is resolvable.
@pytest.mark.parametrize(
    "deliver_kind",
    [k for k, (_, targets_owner) in DELIVERS.items() if targets_owner],
)
def test_unresolvable_home_fails_closed(deliver_kind):
    from cron.delivery_guard import check_cross_identity_delivery

    deliver, _ = DELIVERS[deliver_kind]
    with patch("cron.scheduler_delivery._get_home_target_chat_id", return_value=""):
        err = check_cross_identity_delivery(REQUESTERS["contact-wa"], deliver)
    assert err is not None, (
        f"deliver={deliver_kind} ({deliver!r}) with NO resolvable home was ALLOWED "
        f"for a non-owner — guard failed open"
    )


@pytest.mark.parametrize(
    "deliver_kind",
    [k for k, (_, targets_owner) in DELIVERS.items() if not targets_owner],
)
def test_unresolvable_home_still_allows_non_home_targets(deliver_kind):
    """Fail-closed must not become fail-everything: origin/local/bot-chat stay allowed."""
    from cron.delivery_guard import check_cross_identity_delivery

    deliver, _ = DELIVERS[deliver_kind]
    if deliver_kind in ("contact-wa-explicit", "shared-group"):
        pytest.skip("explicit chat needs a home id to prove it is NOT the owner; blocks by design")
    with patch("cron.scheduler_delivery._get_home_target_chat_id", return_value=""):
        err = check_cross_identity_delivery(REQUESTERS["contact-wa"], deliver)
    assert err is None, f"deliver={deliver_kind} ({deliver!r}) wrongly blocked: {err}"


# --- E2E through the real tool paths -------------------------------------


@pytest.fixture
def temp_cron_home(tmp_path):
    from cron import jobs as cron_jobs

    with cron_jobs.use_cron_store(tmp_path):
        cron_jobs.ensure_dirs()
        yield tmp_path


def _chat_session(platform, chat_id, chat_name):
    return set_session_vars(platform=platform, chat_id=chat_id, chat_name=chat_name)


def _create(deliver=None, **kw):
    from tools.cronjob_tools import cronjob

    return json.loads(
        cronjob(action="create", schedule="every 4h", prompt="watch the thing", deliver=deliver, **kw)
    )


class TestCreatePath:
    def test_contact_to_owner_home_is_refused_and_nothing_persisted(self, temp_cron_home):
        from cron.jobs import list_jobs

        tokens = _chat_session("whatsapp", CONTACT_WA_LID, "Quetzalcoatl")
        try:
            res = _create(deliver=f"whatsapp:{OWNER_WA_PHONE}")
        finally:
            clear_session_vars(tokens)
        assert res["success"] is False
        assert "Cross-identity delivery blocked" in res["error"]
        assert list_jobs(include_disabled=True) == []

    def test_contact_failure_route_to_owner_is_refused(self, temp_cron_home):
        """Failure notices are outbound delivery too; they must obey the same boundary."""
        from cron.jobs import list_jobs

        tokens = _chat_session("whatsapp", CONTACT_WA_LID, "Quetzalcoatl")
        try:
            res = _create(deliver="origin", failure_deliver="telegram")
        finally:
            clear_session_vars(tokens)
        assert res["success"] is False
        assert "Cross-identity delivery blocked" in res["error"]
        assert list_jobs(include_disabled=True) == []

    def test_contact_omitted_deliver_is_allowed(self, temp_cron_home):
        tokens = _chat_session("whatsapp", CONTACT_WA_LID, "Quetzalcoatl")
        try:
            res = _create()
        finally:
            clear_session_vars(tokens)
        assert res["success"] is True
        from cron.jobs import get_job

        assert get_job(res["job_id"])["origin"]["chat_id"] == CONTACT_WA_LID

    def test_contact_to_shared_group_is_allowed(self, temp_cron_home):
        """Contact → shared family group: a contact may deliver elsewhere than the owner."""
        tokens = _chat_session("whatsapp", "80000000000003@lid", "Contact")
        try:
            res = _create(deliver=f"whatsapp:{SHARED_GROUP}")
        finally:
            clear_session_vars(tokens)
        assert res["success"] is True, res

    def test_owner_to_own_home_is_allowed(self, temp_cron_home):
        tokens = _chat_session("whatsapp", OWNER_WA_LID, "Owner")
        try:
            res = _create(deliver="telegram")
        finally:
            clear_session_vars(tokens)
        assert res["success"] is True, res

    def test_no_session_origin_is_allowed(self, temp_cron_home):
        """CLI/TUI create: no origin, no verdict."""
        res = _create(deliver="telegram")
        assert res["success"] is True, res


class TestUpdatePath:
    def test_contact_job_cannot_be_repointed_at_owner(self, temp_cron_home):
        from cron.jobs import get_job
        from tools.cronjob_tools import cronjob

        tokens = _chat_session("whatsapp", CONTACT_WA_LID, "Quetzalcoatl")
        try:
            created = _create()
        finally:
            clear_session_vars(tokens)
        assert created["success"] is True
        res = json.loads(cronjob(action="update", job_id=created["job_id"], deliver="telegram"))
        assert res["success"] is False
        assert "Cross-identity delivery blocked" in res["error"]
        assert get_job(created["job_id"])["deliver"] != "telegram"

    def test_contact_job_failure_route_cannot_be_repointed_at_owner(self, temp_cron_home):
        from cron.jobs import get_job
        from tools.cronjob_tools import cronjob

        tokens = _chat_session("whatsapp", CONTACT_WA_LID, "Quetzalcoatl")
        try:
            created = _create(deliver="origin")
        finally:
            clear_session_vars(tokens)
        assert created["success"] is True
        res = json.loads(
            cronjob(action="update", job_id=created["job_id"], failure_deliver="telegram")
        )
        assert res["success"] is False
        assert "Cross-identity delivery blocked" in res["error"]
        assert not get_job(created["job_id"]).get("failure_deliver")

    def test_contact_job_can_be_repointed_at_contact(self, temp_cron_home):
        from cron.jobs import get_job
        from tools.cronjob_tools import cronjob

        tokens = _chat_session("whatsapp", CONTACT_WA_LID, "Quetzalcoatl")
        try:
            created = _create(deliver=f"whatsapp:{SHARED_GROUP}")
        finally:
            clear_session_vars(tokens)
        res = json.loads(
            cronjob(action="update", job_id=created["job_id"], deliver=f"whatsapp:{CONTACT_WA_LID}")
        )
        assert res["success"] is True, res
        assert get_job(created["job_id"])["deliver"] == f"whatsapp:{CONTACT_WA_LID}"


# --- literal-token bypass: friendly names / handles that RESOLVE to owner ----
#
# Reviewer gap (2026-09-21): the explicit ``platform:chat`` form was compared
# LITERALLY against the owner home id. But at fire time the scheduler runs the
# token through ``resolve_send_target`` (plugin parser → channel directory), so
# a non-owner could deliver to ``telegram:Bot Home`` — a friendly directory name
# that is NOT literally the owner id yet resolves to the owner's home channel —
# and slip past the guard while still landing on the owner. These tests pin the
# fix: the guard must resolve the token through the SAME resolver before
# comparing, and fail closed if resolution is impossible or ambiguous.
#
# The owner's telegram home directory name maps to OWNER_TG; an ordinary
# non-owner name maps to a different chat. ``home-alias`` is a second friendly
# label for the same owner home channel (channel directories routinely carry
# aliases), standing in for the "handle" form that also resolves to owner.
_DIR_TG = {
    "bot home": OWNER_TG,          # the owner's home channel, by friendly name
    "home-alias": OWNER_TG,        # a second directory label for the same channel
    "family plants": CONTACT_TG,   # an ordinary non-owner destination
}


def _fake_channel_dir(platform: str, name: str):
    if platform.lower() != "telegram":
        return None
    return _DIR_TG.get(name.strip().lstrip("#").lower())


class TestFriendlyNameResolutionBypass:
    """A directory name/handle that resolves to owner home is an owner target."""

    def test_contact_to_owner_home_by_directory_name_is_blocked(self):
        from cron.delivery_guard import check_cross_identity_delivery

        with patch("gateway.channel_directory.resolve_channel_name", side_effect=_fake_channel_dir):
            err = check_cross_identity_delivery(REQUESTERS["contact-tg"], "telegram:Bot Home")
        assert err is not None, (
            "a friendly directory name resolving to the owner home was ALLOWED for a "
            "non-owner — literal-token bypass still open"
        )
        assert "Cross-identity delivery blocked" in err

    def test_contact_to_owner_home_by_alias_name_is_blocked(self):
        """A second directory label for the owner home channel must also block."""
        from cron.delivery_guard import check_cross_identity_delivery

        with patch("gateway.channel_directory.resolve_channel_name", side_effect=_fake_channel_dir):
            err = check_cross_identity_delivery(REQUESTERS["contact-tg"], "telegram:home-alias")
        assert err is not None, "an alias directory name resolving to owner home was ALLOWED"

    def test_contact_owner_home_name_in_combo_taints(self):
        """One friendly-name element resolving to owner home taints a combo deliver."""
        from cron.delivery_guard import check_cross_identity_delivery

        with patch("gateway.channel_directory.resolve_channel_name", side_effect=_fake_channel_dir):
            err = check_cross_identity_delivery(
                REQUESTERS["contact-tg"], "origin,telegram:Bot Home"
            )
        assert err is not None, "a combo with a name resolving to owner home was ALLOWED"

    # --- controls: the fix must not become fail-everything ---

    def test_contact_to_ordinary_named_destination_is_allowed(self):
        """A friendly name resolving to a NON-owner chat stays deliverable."""
        from cron.delivery_guard import check_cross_identity_delivery

        with patch("gateway.channel_directory.resolve_channel_name", side_effect=_fake_channel_dir):
            err = check_cross_identity_delivery(REQUESTERS["contact-tg"], "telegram:Family Plants")
        assert err is None, f"an ordinary non-owner named destination was wrongly blocked: {err}"

    def test_contact_to_unknown_name_on_builtin_is_allowed(self):
        """Unknown name on a built-in platform passes through unresolved → not owner-addressed.

        This is a genuine non-owner destination the adapter validates at send time,
        not the owner home; blocking it would be fail-everything, not fail-closed.
        """
        from cron.delivery_guard import check_cross_identity_delivery

        with patch("gateway.channel_directory.resolve_channel_name", side_effect=_fake_channel_dir):
            err = check_cross_identity_delivery(REQUESTERS["contact-tg"], "telegram:No Such Channel")
        assert err is None, f"unknown builtin destination wrongly blocked: {err}"

    def test_owner_to_owner_home_by_directory_name_is_allowed(self):
        """The owner may deliver to their own home by friendly name."""
        from cron.delivery_guard import check_cross_identity_delivery

        with patch("gateway.channel_directory.resolve_channel_name", side_effect=_fake_channel_dir):
            err = check_cross_identity_delivery(REQUESTERS["owner-tg"], "telegram:Bot Home")
        assert err is None, f"owner delivering to own home by name was blocked: {err}"

    # --- fail closed on resolver failure / malformed output ---

    def test_resolver_exception_fails_closed(self):
        """If the resolver raises, a non-owner explicit target must be treated as owner."""
        from cron.delivery_guard import check_cross_identity_delivery

        def _boom(*a, **k):
            raise RuntimeError("directory unavailable")

        with patch("tools.send_message_tool.resolve_send_target", side_effect=_boom):
            err = check_cross_identity_delivery(REQUESTERS["contact-tg"], "telegram:Bot Home")
        assert err is not None, "resolver exception did not fail closed"

    def test_resolver_error_return_fails_closed(self):
        """A resolution error (could not turn token into an id) must fail closed."""
        from cron.delivery_guard import check_cross_identity_delivery

        with patch(
            "tools.send_message_tool.resolve_send_target",
            return_value=(None, None, "Could not resolve 'x' on telegram."),
        ):
            err = check_cross_identity_delivery(REQUESTERS["contact-tg"], "telegram:Bot Home")
        assert err is not None, "resolver error return did not fail closed"

    def test_resolver_malformed_output_fails_closed(self):
        """A non-string / empty resolved chat id must fail closed, not pass."""
        from cron.delivery_guard import check_cross_identity_delivery

        with patch(
            "tools.send_message_tool.resolve_send_target",
            return_value=(12345, None, None),  # non-string chat id
        ):
            err = check_cross_identity_delivery(REQUESTERS["contact-tg"], "telegram:Bot Home")
        assert err is not None, "malformed resolver output did not fail closed"

    def test_resolver_clean_none_leaves_literal_verdict(self):
        """When the resolver cleanly declines (chat_id=None, no error), the literal
        verdict stands — a non-owner named destination that does not match home
        literally is allowed."""
        from cron.delivery_guard import check_cross_identity_delivery

        with patch(
            "tools.send_message_tool.resolve_send_target",
            return_value=(None, None, None),
        ):
            err = check_cross_identity_delivery(REQUESTERS["contact-tg"], "telegram:Family Plants")
        assert err is None, f"clean-unresolved non-owner target wrongly blocked: {err}"

    def test_e2e_create_contact_to_owner_home_by_name_is_refused(self, temp_cron_home):
        """End-to-end through cronjob(): a contact cannot schedule delivery to the
        owner's home channel via a friendly directory name."""
        from cron.jobs import list_jobs

        tokens = _chat_session("telegram", CONTACT_TG, "Someone")
        try:
            with patch(
                "gateway.channel_directory.resolve_channel_name", side_effect=_fake_channel_dir
            ):
                res = _create(deliver="telegram:Bot Home")
        finally:
            clear_session_vars(tokens)
        assert res["success"] is False
        assert "Cross-identity delivery blocked" in res["error"]
        assert list_jobs(include_disabled=True) == []
