"""Third-person-recipient guard: block owner-report register leaking into a
contact's DM. The 2026-08-22 Jordan incident is the canonical case."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from gateway.response_filters import mentions_recipient_third_person as leaks


def test_real_leak_oscar():
    # The exact message Jordan received.
    assert leaks("Transcode probado en vivo: sano. El problema de Malcolm es que "
                 "esos episodios la tele de Jordan no reproduce directo. Le digo que "
                 "intente de nuevo.", "Jordan")
    assert leaks("la TV de Jordan no los reproduce directo", "Jordan")
    assert leaks("Jordan no puede reproducir eso, hay que transcodificar", "Jordan")
    assert leaks("ya le avisé a Jordan que quedó", "Jordan")
    assert leaks("la cuenta de Jordan está impagada desde junio", "Jordan Perez")  # surname ignored


def test_legit_vocative_not_flagged():
    assert not leaks("Jordan, ya lo revisé 👍", "Jordan")
    assert not leaks("¡Hola Jordan! 😄 ¿Qué pasó?", "Jordan")
    assert not leaks("Va, Jordan — dame un momento.", "Jordan")
    assert not leaks("Ya quedó, Jordan. Dale de nuevo.", "Jordan")
    assert not leaks("Listo Jordan 🙌 disfruta la peli", "Jordan")
    assert not leaks("Perdón Jordan, se me volvió a colar.", "Jordan")
    assert not leaks("Gracias Jordan, con gusto.", "Jordan")
    assert not leaks("Buenas noches Jordan, todo en orden.", "Jordan")
    assert not leaks("No hay problema. Disfruta.", "Jordan")  # name absent
    assert not leaks("nada 👍", "Jordan")


def test_edge_cases():
    assert not leaks("", "Jordan")
    assert not leaks("algo", "")
    assert not leaks("un texto cualquiera", "Al")       # 2-char name: skip
    assert not leaks(None, "Jordan")
    assert not leaks("texto", None)
    # Vocative AND third-person, NO 2nd-person address → still flagged (leak half wins)
    assert leaks("Jordan, ya revisé. La tele de Jordan necesita transcode.", "Jordan")


# ---------------------------------------------------------------------------
# THE CLASS GATE (2026-09-05) — the guard must never decide by recognising the
# GREETING. 2026-09-05 false positive: "Holaaa Robin 👋 ¿Qué se cuenta?" was
# held as an owner-report. Identical register to "Hola Robin, ¿qué se
# cuenta?", which passed. Two things differed and both were cosmetic: the
# greeting was elongated ("Holaaa" is not in the greet allow-list) and the
# separator after the name was an emoji, not a comma. Same shape as the
# 2026-08-31 Markdown incident — an ANCHOR-based vocative stripper that only
# recognises the shapes someone enumerated. Greetings/interjections/separators
# are an OPEN set (Holaaa, Órale, Ey, Jajaja, Buenaaas, 👋, 🙌, !!); the
# third-person FRAME the name sits in is a CLOSED set (preposition/article
# before the name, a clitic or 3rd-person verb after it). A correct guard keys
# on the closed set. These matrices demand it: no greeting/separator may flip a
# vocative into a leak, and no greeting may launder a real leak.
# ---------------------------------------------------------------------------
GREETINGS = (
    "Hola", "Holaaa", "Holaaaaa", "hola", "HOLA", "Holi", "Holis", "Buenas",
    "Buenaaas", "Buenas noches", "Qué tal", "Que tal", "Órale", "Orale", "Ey",
    "Eyyy", "Hey", "Oye", "Jajaja", "Jaja", "Va", "Listo", "Sale", "Ándale",
    "Andale", "Gracias", "Perdón", "Simón", "Nel", "Uy", "Ay", "Ahh", "Mmm",
    "Ok", "Okey", "Vale", "Claro", "Bueno", "Pues", "Ya", "Bien", "Chido",
    "Feliz cumple", "Saludos", "Buen día", "Wey", "Güey", "Carnal", "Compa",
    "Mano", "Hermano", "Amigo", "Bro", "Jefe",
)
SEPARATORS = (
    ",", "!", "!!", "!!!", "?", ".", "…", " —", " -", ":", ";", "",
    " 👋", " 👋🏽", " 🙌", " 😄", " 🤝", " 🙏", "!! 🎉", " 😂😂", " 🫡", " 🔥🔥",
    " ✅", " 👀", " 💪", " 👌", " ✌️", " 😉", " ❤️", " 🎂", " ✨", " 👋👋👋",
)
# Tails with NO second-person marker — the fallback path is the only defence.
TAILS = (
    "¿Qué se cuenta?", "¿qué se cuenta?", "ya quedó.", "todo en orden.",
    "¿cómo va todo?", "aquí andamos.", "sin novedad por acá.", "ya está listo",
    "mañana lo revisamos.", "cuando pueda.", "¿todo bien?", "va que va.",
    "", "🙌", "😄", "¿Qué onda?", "¿qué hay?", "buenas noches!", "ahí quedó",
    "ya salió el video", "la peli ya está", "¿Qué pasó?", "¿Cómo estás?",
)


def test_vocative_verdict_is_invariant_under_greeting_and_separator():
    """No greeting × separator × tail combination may hold a vocative reply."""
    name = "Robin"
    broken = []
    for g in GREETINGS:
        for sep in SEPARATORS:
            for tail in TAILS:
                for msg in (f"{g} {name}{sep} {tail}".strip(),
                            f"{name}{sep} {tail}".strip(),
                            f"{g}, {name}{sep} {tail}".strip(),
                            f"{tail} {g} {name}{sep}".strip(),
                            f"{tail}{sep or ','} {name}{sep}".strip()):
                    if leaks(msg, "Robin López"):
                        broken.append(repr(msg))
    assert not broken, (
        f"{len(broken)} vocative replies held as leaks — the guard is still keyed "
        "on a greeting/separator allow-list:\n  " + "\n  ".join(broken[:40])
    )


def test_exact_2026_09_05_false_positive():
    assert not leaks("Holaaa Robin 👋 ¿Qué se cuenta?", "Robin López")
    assert not leaks("Holaaa Robin 👋 ¿Qué se cuenta?", "Robin")
    assert not leaks("¡Robin! ¿Qué onda?", "Robin López")
    assert not leaks("Buenas Robin 🙌 ya quedó", "Robin López")
    assert not leaks("Órale Robin, va", "Robin López")
    assert not leaks("Jajaja Robin eres un caso", "Robin López")
    assert not leaks("Hola Robin 👋", "Robin López")
    assert not leaks("Robin 👋 ¿todo bien?", "Robin López")
    assert not leaks("Feliz cumple Robin 🎂🎉", "Robin López")
    # Name mid-sentence but still addressed (no frame around it).
    assert not leaks("Qué gusto saludarte Robin", "Robin López")  # 'te' clitic
    assert not leaks("Ya quedó Robin 🙌", "Robin López")
    assert not leaks("Mañana te marco Robin", "Robin López")
    # Comma-less vocative + state/event verb: ambiguous → PASS (precision half).
    assert not leaks("Robin ya quedó 🙌", "Robin López")
    assert not leaks("Robin ya salió la peli", "Robin López")
    assert not leaks("Robin ya está listo el server.", "Robin López")
    # Impersonal idioms with an agent-verb homograph must not count.
    assert not leaks("Robin puede ser que sí, déjame ver.", "Robin López")
    assert not leaks("Robin tiene que ser el cable.", "Robin López")
    assert not leaks("Robin ha sido un placer.", "Robin López")
    assert not leaks("Robin quiere decir que ok.", "Robin López")
    # Bare present forms are tú-imperatives here, not 3rd-person.
    assert not leaks("Robin manda el archivo cuando puedas.", "Robin López")
    assert not leaks("Robin revisa el link porfa.", "Robin López")
    # English vocatives.
    assert not leaks("Thanks Robin!", "Robin López")
    assert not leaks("Hey Robin! all good?", "Robin López")
    assert not leaks("Robin is it working now?", "Robin López")
    assert not leaks("Robin — see the attached PDF.", "Robin López")


def test_owner_report_frames_are_caught():
    """Recall floor: the shapes HAL actually writes in owner reports (mined from
    session history 2026-08/09), with the recipient's name substituted in."""
    must = (
        "Le avisé a Robin.",
        "Le aviso a Robin mientras corre.",
        "Reporto el estado real a Robin.",
        "El problema puede ser de mi lado, no del servidor de Robin.",
        "Voy al eje donde Robin tiene ventaja real.",
        "Robin me dijo que sí.",
        "Robin ha estado insistiendo.",
        "Robin quiere el enlace hoy.",
        "Robin no contestó todavía.",
        "Hablé con Robin, ya quedó lo suyo.",
        "Para Robin el acceso ya quedó.",
        "Pendiente: responderle a Robin con el PDF.",
        "Según Robin el archivo no abre.",
        "This is Robin's DM, so the right place is his own memory silo.",
        "Robin already has a proper silo.",
        "Robin specifically wants a small image per term.",
        "Reply to Robin first, then escalate to Sam.",
        "Then confirm to Robin.",
        "…then notifies Robin when done.",
        "Notified Robin.",
        "I told Robin to retry.",
        "Robin asked for the video; sending it tomorrow.",
        "Robin is waiting on the link.",
        "Sent Robin the invoice.",
        "Let Robin know it's done.",
        "Ping Robin once the file lands.",
        "Robin hasn't replied yet.",
        "Robin apparently didn't get the code.",
        "Waiting on Robin's confirmation.",
        "Updated Robin on the status.",
    )
    missed = [t for t in must if not leaks(t, "Robin López")]
    assert not missed, "owner-report frames passed the guard:\n  " + "\n  ".join(missed)


def test_greeting_does_not_launder_a_real_leak():
    """A greeting in front cannot turn owner-report narration into a reply."""
    leaks_ = (
        "la tele de Robin no reproduce directo.",
        "la cuenta de Robin sigue impagada desde junio.",
        "Le digo a Robin que lo intente de nuevo.",
        "Robin pidió el video; se lo mando mañana.",
        "Robin no puede reproducir eso, hay que transcodificar.",
        "ya le avisé a Robin que quedó.",
        "Robin dice que no le llega el código.",
        "El acceso de Robin ya está activo; no requiere acción de mi parte.",
        "Confirmado con Robin: su cuenta sigue impagada.",
        "Robin tiene el video pendiente, se lo mando mañana.",
        "Le mandé a Robin el enlace y ya lo abrió.",
        "Robin ya recibió el enlace.",
    )
    broken = []
    for body in leaks_:
        for prefix in ("", "Hola. ", "Holaaa 👋 ", "Listo. ", "Reporte: ", "Ok — "):
            msg = f"{prefix}{body}"
            if not leaks(msg, "Robin López"):
                broken.append(repr(msg))
    assert not broken, (
        "real owner-report leaks passed the guard:\n  " + "\n  ".join(broken)
    )
    # Vocative + leak in the same message: leak half wins (unchanged contract).
    assert leaks("Hola Robin. La tele de Robin no reproduce.", "Robin López")
    assert leaks("Holaaa Robin 👋 Le digo a Robin que lo intente.", "Robin López")


def test_second_person_address_is_never_a_leak():
    # Root cause: a reply that addresses the reader in 2nd person is not an
    # owner-report, even when it names the contact in a 3rd-person-looking
    # clause. The Casey false positive that motivated the fix:
    assert not leaks("Directo, Casey, porque tú mismo pusiste la regla: el 17 de "
                     "agosto me dijiste cero LoL. ¿Ya terminaste el curso o no?", "Casey Robin Graciano L")
    assert not leaks("Jordan, tu cuenta ya quedó al corriente, dale de nuevo.", "Jordan")
    assert not leaks("Casey, ¿ya terminaste el curso?", "Casey")
    assert not leaks("Your Chill account is active — enjoy, Jordan.", "Jordan")
    assert not leaks("te reactivé el acceso, Jordan", "Jordan")
    # The clitic guard must not match inside words: "intente" is not "te".
    assert leaks("Le digo a Jordan que lo intente de nuevo.", "Jordan")


def test_markdown_emphasis_around_vocative_is_not_a_leak():
    """A vocative wrapped in Markdown emphasis is still a vocative.

    2026-08-31 false positive: a research handoff opening with a bolded
    vocative — ``**Casey — la investigación ya cerró...**`` — was held as an
    owner-report leak. The vocative pattern anchors the name to a clause
    boundary (start-of-string / sentence punctuation), and the leading ``**``
    sits between the anchor and the name, so the strip never fired and the
    name survived into the third-person residue. The identical text without
    the asterisks passed. Markdown is cosmetic; the guard must see through it
    (same bug class as the MEDIA-tag emphasis tolerance in
    gateway/platforms/base.py).
    """
    long_name = "Casey Robin Graciano L"
    # The exact shape that was wrongly held.
    assert not leaks("**Casey — la investigación jurídica ya cerró y trae dos "
                     "hallazgos que cambian el caso.**", long_name)
    # Every emphasis flavour, leading and trailing.
    assert not leaks("*Casey — ya quedó.*", long_name)
    assert not leaks("__Casey__: ya quedó.", long_name)
    assert not leaks("~Casey~, ya quedó.", long_name)
    assert not leaks("**Hola Jordan** — ya quedó.", "Jordan")
    assert not leaks("Ya quedó, **Jordan**.", "Jordan")
    assert not leaks("Listo, *Jordan*", "Jordan")

    # ...and emphasis must NOT become a laundering channel for a real leak.
    assert leaks("**la TV de Jordan no reproduce directo**", "Jordan")
    assert leaks("Transcode sano. *La cuenta de Jordan sigue impagada.*", "Jordan")
    assert leaks("**Casey**, ya revisé. La mamá de **Casey** debe radicar hoy.", long_name)


def test_emphasis_stripping_preserves_second_person_detection():
    # The same normalisation must not break the 2nd-person discriminator:
    # a bolded pronoun still addresses the reader.
    assert not leaks("**Casey**, porque **tú** mismo pusiste la regla.", "Casey")
    assert not leaks("Jordan, __tu__ cuenta ya quedó.", "Jordan")
    # snake_case identifiers must survive normalisation as words (no false
    # vocative/name matches manufactured by gluing tokens together).
    assert leaks("El job de Jordan (send_media_report) sigue fallando.", "Jordan")


def test_emphasis_normalisation_is_edge_symmetric():
    """``__Casey__`` must be stripped on BOTH sides, not just the leading one.

    Guards a real trap: an earlier draft left a trailing underscore glued to
    the name (``Casey_``) and the test still passed — because ``_`` is a regex
    word character, so ``\\bCasey\\b`` simply failed to match the residue. The
    verdict was right for the wrong reason and would have broken the moment the
    residue test changed. Assert the normalisation itself, not just its effect.
    """
    from gateway.response_filters import _EMPHASIS_RUN as E

    def norm(s):  # emphasis-stripped, whitespace-collapsed
        return " ".join(E.sub(" ", s).split())

    assert norm("__Casey__: ya quedó.") == "Casey : ya quedó."
    assert norm("**Casey — ya.**") == "Casey — ya."
    assert norm("~~Jordan~~") == "Jordan"
    # The point of the test: NO underscore survives glued to the name on
    # either edge (a trailing "Casey_" would only pass by \b accident).
    assert "_" not in norm("__Casey__: ya quedó.")
    # Identifier underscores are word separators, not emphasis — left intact.
    assert E.sub(" ", "send_media_report") == "send_media_report"
    assert E.sub(" ", "a_b_c_d") == "a_b_c_d"
    # No token is ever glued to its neighbour by the substitution.
    assert "CaseyRobin" not in E.sub(" ", "**Casey**Robin")


# ---------------------------------------------------------------------------
# THE CLASS GATE — cosmetic markup must never change a verdict.
#
# Each wrapper takes a plain-prose message and returns the SAME message dressed
# in one Markdown shape. The matrix below asserts every guard returns the same
# verdict for the dressed and undressed text. Adding a row here is how a newly
# discovered shape becomes impossible: the test demands markdown_to_prose()
# handle it, instead of someone patching one more character into an anchor.
#
# Why a matrix and not a list of examples: the 2026-08-31 fix added exactly one
# example (bold) and shipped believing the bug was dead. Five shapes were still
# broken (headings, blockquotes, bullets, links, quote+bold). A per-case test
# cannot fail for a shape nobody thought of; a matrix over wrappers x cases
# fails the moment a wrapper is added.
# ---------------------------------------------------------------------------
COSMETIC_WRAPPERS = (
    ("plain",          lambda s: s),
    ("bold",           lambda s: f"**{s}**"),
    ("italic",         lambda s: f"*{s}*"),
    ("underscore-em",  lambda s: f"__{s}__"),
    ("strikethrough",  lambda s: f"~~{s}~~"),
    ("inline-code",    lambda s: f"`{s}`"),
    ("h2",             lambda s: f"## {s}"),
    ("h3",             lambda s: f"### {s}"),
    ("blockquote",     lambda s: f"> {s}"),
    ("nested-quote",   lambda s: f">> {s}"),
    ("quote+bold",     lambda s: f"> **{s}**"),
    ("dash-bullet",    lambda s: f"- {s}"),
    ("star-bullet",    lambda s: f"* {s}"),
    ("plus-bullet",    lambda s: f"+ {s}"),
    ("ordered-dot",    lambda s: f"1. {s}"),
    ("ordered-paren",  lambda s: f"1) {s}"),
    ("hr-above",       lambda s: f"---\n{s}"),
    ("heading-above",  lambda s: f"## Resumen\n\n{s}"),
    ("bold-heading",   lambda s: f"**Resumen**\n\n{s}"),
)

# (label, message, expected verdict as PLAIN prose)
_VERDICT_CASES = (
    ("vocative-dash",     "Casey — la investigación jurídica ya cerró.",        False),
    ("vocative-comma",    "Casey, ya quedó todo.",                              False),
    ("vocative-greeting", "Hola Jordan, ya quedó.",                              False),
    ("trailing-vocative", "Ya quedó, Jordan.",                                   False),
    ("no-name",           "Ya quedó. Disfruta.",                                False),
    ("real-leak",         "la TV de Jordan no reproduce directo",                True),
    ("real-leak-2",       "la cuenta de Jordan sigue impagada desde junio",      True),
    ("real-leak-3",       "Le digo a Jordan que lo intente de nuevo.",           True),
)


def test_verdicts_are_invariant_under_cosmetic_markup():
    """No Markdown wrapper may flip any verdict. This is the class gate."""
    broken = []
    for case_label, msg, expected in _VERDICT_CASES:
        recipient = "Jordan" if "Jordan" in msg else "Casey Robin Graciano L"
        for wrap_label, wrap in COSMETIC_WRAPPERS:
            got = leaks(wrap(msg), recipient)
            if got != expected:
                broken.append(
                    f"{case_label} + {wrap_label}: expected {expected}, got {got} "
                    f"-> {wrap(msg)!r}"
                )
    assert not broken, (
        "cosmetic markup changed a guard verdict — markdown_to_prose() does not "
        "handle these shapes:\n  " + "\n  ".join(broken)
    )


def test_second_person_detection_is_invariant_under_cosmetic_markup():
    """The 2nd-person discriminator runs on the same normalised prose."""
    from gateway.response_filters import addresses_recipient_second_person as addr

    for msg in ("Casey, porque tú mismo pusiste la regla.",
                "Jordan, tu cuenta ya quedó al corriente.",
                "your account is active"):
        for wrap_label, wrap in COSMETIC_WRAPPERS:
            assert addr(wrap(msg)), f"2nd-person address lost under {wrap_label}: {wrap(msg)!r}"


def test_markdown_to_prose_never_glues_tokens_or_eats_prose():
    """Normalisation is space-substituting and prose-preserving, not deleting."""
    from gateway.response_filters import markdown_to_prose as p

    # Never glue two tokens into a word that was not in the original.
    assert "CaseyRobin" not in p("**Casey**Robin")
    # Inline dashes/hyphens are PROSE, not list markers — must survive.
    assert "—" in p("Casey — la investigación ya cerró.")
    assert "e-mail" in p("mándame un e-mail")
    # Identifiers keep their underscores.
    assert "send_media_report" in p("el job send_media_report falló")
    # Link labels survive; targets do not leak into the prose.
    assert "Casey" in p("[Casey](https://example.com/x)")
    assert "example.com" not in p("[Casey](https://example.com/x)")
    # A real leak stays detectable after normalisation (no content eaten).
    assert "Jordan" in p("> **la TV de Jordan no reproduce**")


def test_guard_call_site_is_dm_gated():
    # The guard flags legit GROUP answers (Riley 2026-08-23): in a group,
    # source.user_name is just the last sender and third-personing a participant
    # is the normal register, not a leak. The call site in run.py MUST gate on
    # chat_type == "dm" so the guard's scope matches the DM-only hazard. This is
    # a structural gate: it fails if someone reintroduces an ungated call.
    #
    # Window-based, not same-line: the condition is legitimately multi-line
    # (it also carries the delivery-bound precondition below), so requiring
    # chat_type on the guard's own line would fail on correct code.
    import pathlib
    run_py = pathlib.Path(__file__).resolve().parents[1] / "gateway" / "run_turn.py"
    lines = run_py.read_text().splitlines()
    idx = [i for i, ln in enumerate(lines)
           if "mentions_recipient_third_person(" in ln
           and "import" not in ln and "def " not in ln]
    assert idx, "guard call site not found — did the symbol move?"
    for i in idx:
        window = "\n".join(lines[max(0, i - 6):i + 1])
        assert 'chat_type' in window and '"dm"' in window, (
            f"ungated guard call (must require chat_type == 'dm') at line {i + 1}: "
            f"{lines[i].strip()}")


def test_alarming_guard_is_delivery_bound():
    """The reroute-and-alarm guard must only judge text that will be SENT.

    2026-09-01 false alarm (session 20260901_023948_5d503b3f): the turn had
    already replied to the contact via send_message(target="current"). The
    reply-gate dedup then suppresses the model's free-text tail as a duplicate
    — and that tail is exactly where the model writes its owner-facing wrap-up
    ("Le avisé a Alex"). Third-person register there is CORRECT. The guard
    nonetheless "held" the never-to-be-delivered tail and told the owner a
    reply had been withheld from the contact. Nothing had been withheld; she
    had already received her real reply 17 seconds earlier.

    This guard REROUTES and ANNOUNCES, so a false positive is not a harmless
    no-op like its suppress-only siblings — it makes a false claim to the
    owner, on exactly the turns that went well. It must therefore be gated on
    the delivery decision (`_delivered_via_tool`), not just on register.

    Structural: fails if someone reintroduces a delivery-unbound alarm call.
    """
    import pathlib
    run_py = pathlib.Path(__file__).resolve().parents[1] / "gateway" / "run_turn.py"
    lines = run_py.read_text().splitlines()

    idx = [i for i, ln in enumerate(lines)
           if "mentions_recipient_third_person(" in ln
           and "import" not in ln and "def " not in ln]
    assert idx, "guard call site not found — did the symbol move?"

    for i in idx:
        # The precondition may sit a few lines above in a multi-line condition.
        window = "\n".join(lines[max(0, i - 6):i + 1])
        assert "_tail_reaches_recipient" in window or "_delivered_via_tool" in window, (
            "third-person guard call is NOT delivery-bound — it can alarm the "
            "owner about a tail the reply-gate was going to suppress anyway "
            f"(line {i + 1}): {lines[i].strip()}"
        )

    # And the precondition must be derived from the delivery signal, not
    # hardcoded True (which would silently restore the old behaviour).
    src = "\n".join(lines)
    assert "_tail_reaches_recipient = not _delivered_via_tool" in src, (
        "_tail_reaches_recipient must be derived from _delivered_via_tool "
        "(the transcript-authoritative dedup signal), not set to a constant"
    )


def test_judge_gates_every_non_owner_dm_final_before_send():
    """The semantic judge is the GATE, not a post-hoc detector (2026-09-12).

    History: the judge used to run on the hot path only when a regex fired,
    and after delivery otherwise. The regexes are blind by construction to
    the shapes that actually leak (a pushname "Hey!" instead of a first
    name; prose status reports with no checkmark bullets), so on 2026-09-11
    three owner-register messages reached a contact's DM and the judge —
    which holds all three unanimously — only ever saw them AFTER the send.

    Structural invariants, asserted on the source because either one
    disappearing silently restores the old architecture:

    1. The judge call is NOT nested under a regex condition: the branch that
       awaits judge_delivery is gated on delivery-reaching-recipient + DM +
       non-empty response, and nothing else.
    2. It IS awaited (a hold must happen before the send).
    3. No post-hoc scheduler remains as a second, weaker path.
    """
    import pathlib
    import re
    run_py = pathlib.Path(__file__).resolve().parents[1] / "gateway" / "run_turn.py"
    src = run_py.read_text(encoding="utf-8")

    m = re.search(
        r"if (?P<cond>[^\n]*?):\n\s+from gateway\.delivery_judge import judge_delivery\n", src
    )
    assert m, "judge_delivery is not wired into the non-owner DM final path"
    cond = m.group("cond")
    for must in ("_tail_reaches_recipient", "_is_dm", "response"):
        assert must in cond, f"judge gate must depend on {must}; got: {cond}"
    for must_not in ("_regex_leak", "_name_leak", "_struct_leak"):
        assert must_not not in cond, (
            f"judge gate is conditioned on {must_not}: the regex is back to "
            f"deciding WHEN the judge runs, which is the recall hole this test exists for"
        )

    after = src[m.end(): m.end() + 800]
    assert "await" in after and "judge_delivery(" in after, (
        "judge_delivery must be awaited on the delivery path so a hold happens before the send"
    )

    assert "_schedule_register_audit(" not in src.replace("# _schedule_register_audit", ""), (
        "a post-delivery register audit still exists — the judge on the hot path "
        "made it redundant; a second weaker path is where the next leak hides"
    )


def test_platform_notice_rail_routes_by_reader():
    """Operator notices (subagent timeout, credit band, no-home nudge) never
    reach a non-owner DM (Jordan, 2026-09-11: '⚠️ Subagent timed out — Add the
    movie Uptown Girls…: Timed out after 1200s'). Routed at the RAIL, so every
    caller is covered at once."""
    import pathlib
    import re
    run_py = pathlib.Path(__file__).resolve().parents[1] / "gateway" / "run_notifications.py"
    src = run_py.read_text(encoding="utf-8")
    m = re.search(r"async def _deliver_platform_notice\(self, source, content: str\) -> None:\n(.*?)\n        adapter = self\._(?:adapter_for_source|delivery_adapter_for)\(source\)", src, flags=re.S)
    assert m, "_deliver_platform_notice not found"
    head = m.group(1)
    assert "_should_skip_user_profile_for_source(source=source)" in head and "_notify_owner_home(" in head, (
        "_deliver_platform_notice must reroute non-owner readers to the owner home BEFORE resolving the adapter"
    )

# ---------------------------------------------------------------------------
# Owner-report STRUCTURE guard (2026-09-09) — name-independent leak block.
# Closes the 6/15 blind silos found in the 2026-09-09 audit at once, since it
# keys on report scaffolding rather than the recipient's name.
# ---------------------------------------------------------------------------
from gateway.response_filters import is_owner_report_scaffold as _scaffold


def test_scaffold_blocks_real_oscar_leak():
    # The exact shape delivered to Jordan 2026-09-09: autonomy preamble + bold
    # "Done:" heading + checkmark bullets, NO third-person name anywhere.
    leak = (
        "Everything is set up and running autonomously. Here's where things stand:\n\n"
        "*Done:*\n- ✅ Added Rise of the Guardians (2012) to the library\n"
        "- ✅ Found and started the Latino audio donor"
    )
    assert _scaffold(leak)


def test_scaffold_blocks_name_free_shapes():
    assert _scaffold("*Done:*\n- Added the movie")            # bold label alone
    assert _scaffold("*Hecho:*\n- Agregué la película")       # Spanish label
    assert _scaffold("- ✅ Descargué la base\n- ✅ Encontré el audio")  # 2 checks
    assert _scaffold("Confirmado disponible. No requiere acción adicional de mi parte.")
    assert _scaffold("Todo listo y corriendo de forma autónoma. Te aviso.")
    assert _scaffold("Ya arranqué. Le avisaré cuando termine de descargar.")


def test_scaffold_passes_genuine_replies():
    # Warm replies, single check, file delivery, questions — none are reports.
    assert not _scaffold("Listo Taylor ✅ ya quedó, pruébalo cuando quieras 🌸")
    assert not _scaffold("ya está disponible ✅")
    assert not _scaffold("Aquí está la peli 🎬 1080p, audio latino, dura 1h37. Disfrútala!")
    assert not _scaffold("¿La quieres en 1080p o 4K? Dime y la bajo.")
    assert not _scaffold("Done! The movie is in your library now, Latino audio included. Enjoy 🎬")
    assert not _scaffold("Oye - ya quedó lo que pediste, avísame si algo")
    assert not _scaffold("jajaja va va 😂😂 ahorita lo veo y te digo")
    assert not _scaffold("")


def test_scaffold_passes_teaching_reply_with_correct_incorrect_glyphs():
    # 2026-09-10 Alex false positive: an English-grammar lesson uses ✅/❌ as
    # CORRECT/INCORRECT answer markers (≥2 of them) and reads exactly like a
    # status board — but it addresses the reader in the second person, so it is
    # a genuine teaching reply, never an owner-report. The root-cause gate
    # (second-person address => not a report) must let it through whatever
    # glyphs it carries.
    lesson = (
        "¡Claro, Alex! 😊 regla sencillita:\n"
        '*El pasado de "go" es "went"*\n'
        '✅ She *went* out. → aquí "went" carga el pasado\n'
        '✅ She *didn\'t go* out. → aquí "didn\'t" carga el pasado\n'
        "❌ She *didn't went* out. → ¡doble pasado! No se puede\n"
        "¿Quieres que te ponga unas frasecitas para practicar y me las contestas? 💪"
    )
    assert not _scaffold(lesson)
    # A checklist board with the SAME glyphs but NO second-person address is
    # still a report and must still be blocked — the gate keys on register,
    # not on the glyph count.
    assert _scaffold("✅ Cuenta reactivada\n❌ Pago pendiente")


# ---------------------------------------------------------------------------
# Shape-B matrix (2026-09-11, Alex tuition false positive): the verdict must
# be decided by REGISTER, never by which ambiguous scaffold marker the reply
# happens to carry. The 09-10 fix gated only the ≥2-bullet branch; a
# "*Resumen*" section header over cost bullets — same second-person register
# ("Si quieres te las redacto") — was then held by the bold-label branch.
# Every ambiguous marker × every register, one assertion per cell.
# ---------------------------------------------------------------------------
import pytest as _pytest

_AMBIGUOUS_SCAFFOLDS = {
    "bold-label-es": "*Resumen*\n{body}",
    "bold-label-colon-es": "*Notas:*\n{body}",
    "bold-label-en": "*Summary*\n{body}",
    "bold-label-results": "*Resultados*\n{body}",
    "two-check-bullets": "✅ {body}\n✅ y también esto",
    "check-and-cross": "✅ {body}\n❌ esto no",
    "label-plus-bullets": "*Resumen*\n• {body}\n• segundo punto\n✅ tercero\n✅ cuarto",
}

_REGISTERS = {
    # second person → genuine reply → must PASS whatever scaffold wraps it
    "2p-tu": ("Tu opción más barata es UTC. Si quieres te redacto las preguntas.", False),
    "2p-en": ("Your cheapest option is UTC. Want me to draft the questions for you?", False),
    # third person / no addressee → owner-report → must BLOCK
    "3p-about-contact": ("La cuenta de Ana sigue impagada. Se le envió recordatorio.", True),
    "3p-log": ("Pedido de Jordan procesado. Radarr buscando la película.", True),
    "3p-en": ("Casey's repo was cloned. Tests still need review.", True),
}

# KNOWN LIMIT, pinned on purpose. Formal Spanish "usted" is a second-person
# register whose surface forms (su/le/lleve/quiere) are BYTE-IDENTICAL to
# third-person forms: "su cita" is your appointment or her appointment. A
# word-level regex cannot tell them apart, and widening _SECOND_PERSON_ADDRESS
# with su|le would flip real leaks ("Le recomendé UTC, queda pendiente que
# ella confirme") to PASS. Disambiguation belongs to the async semantic judge
# (gateway/register_audit.py — its primary lane was chosen 2026-09-11 because
# it reads usted as second person). Until the sync guard grows a real usted
# detector, a usted reply under an ambiguous scaffold is HELD, not leaked: the
# fail-safe direction. strict=True so a silent fix or a silent regression to
# this boundary is noticed either way.
_USTED_2P = "Su cita es el 17. Lleve su identificación, por favor. ¿Quiere que le explique?"


@_pytest.mark.parametrize("scaffold_kind", list(_AMBIGUOUS_SCAFFOLDS))
@_pytest.mark.xfail(
    strict=True,
    reason="usted register is regex-indistinguishable from 3rd person; sync guard holds it (fail-safe), async judge decides",
)
def test_scaffold_usted_register_is_a_known_sync_limit(scaffold_kind):
    text = _AMBIGUOUS_SCAFFOLDS[scaffold_kind].format(body=_USTED_2P)
    assert not _scaffold(text)


@_pytest.mark.parametrize("scaffold_kind", list(_AMBIGUOUS_SCAFFOLDS))
@_pytest.mark.parametrize("register_kind", list(_REGISTERS))
def test_scaffold_verdict_is_decided_by_register_not_marker(scaffold_kind, register_kind):
    body, want_block = _REGISTERS[register_kind]
    text = _AMBIGUOUS_SCAFFOLDS[scaffold_kind].format(body=body)
    got = _scaffold(text)
    assert got == want_block, (
        f"cell scaffold={scaffold_kind} register={register_kind}: "
        f"expected {'BLOCK' if want_block else 'PASS'}, got {'BLOCK' if got else 'PASS'} — "
        f"the marker changed the verdict, the register did not"
    )


def test_scaffold_blocks_jessi_tuition_shape_only_when_third_person():
    # The 2026-09-11 held reply, verbatim shape: bold section headers over
    # cost bullets, closing in second person. Must pass. The identical
    # structure narrated ABOUT her must still block.
    second = (
        "Ya con los tres precios, hice la cuenta completa:\n\n"
        "*UNITEC — 3 años*\n• 36 mensualidades × $2,775 = $99,900\n• *Total ≈ $121,000*\n\n"
        "*Resumen*\n• UTC: ~$61k → la más barata.\n• UNITEC: ~$121k.\n\n"
        "Por eso me inclino por *UTC*. Si quieres te las redacto para que se las mandes tal cual."
    )
    third = (
        "Ya con los tres precios, hice la cuenta completa para Alex:\n\n"
        "*Resumen*\n• UTC: ~$61k → la más barata.\n• UNITEC: ~$121k.\n\n"
        "Le recomendé UTC. Queda pendiente que ella confirme."
    )
    assert not _scaffold(second)
    assert _scaffold(third)


def test_scaffold_autonomy_phrase_is_not_excused_by_second_person():
    # The one marker that must NOT be gated: mechanism narration is a leak
    # even when it says "te".
    assert _scaffold("Todo listo y corriendo de forma autónoma. Te aviso cuando termine.")
    assert _scaffold("Ya quedó. No requiere acción de tu parte, te aviso.")


# ---------------------------------------------------------------------------
# Per-silo alias layer (2026-09-09) — re-arms the name guard for silos whose
# display name is a pushname/handle/nickname/emoji-glued name.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    n = 0
    for fn in (test_real_leak_oscar, test_legit_vocative_not_flagged, test_edge_cases,
               test_second_person_address_is_never_a_leak, test_guard_call_site_is_dm_gated,
               test_scaffold_blocks_real_oscar_leak, test_scaffold_blocks_name_free_shapes,
               test_scaffold_passes_genuine_replies):
        fn(); n += 1
        print(f"  {fn.__name__} OK")
    print(f"third-person-recipient guard: {n} groups passed")
