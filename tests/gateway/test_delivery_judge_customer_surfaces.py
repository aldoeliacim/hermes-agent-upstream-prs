"""Customer-facing surfaces are not backend substance.

Ground truth: a *customer* asked how to log in. Several phrasings of the same
login guide were HELD on ``backend_substance_disclosure`` because they named the
client app next to the public address. A first fix keyed on onboarding VERBS and
was instance-fitted: the next phrasing was still held.

Root cause: the customer's own client app and request page were filed with ops
internals (Sonarr/Radarr/qBittorrent/systemd), and the brand name counted as
"deployment context". Customer surfaces are only a leak alongside OPERATOR state
(versions, logs, HTTP codes, refresh/restart, admin plugins, first-person "our
server"). Ops tools stay a leak next to any deployment context.
"""
import pytest

from gateway.delivery_judge import deterministic_privacy_leak_reason as reason


# Customer login/support guidance: names the CLIENT app + PUBLIC host, no
# operator state -> delivered.
LOGIN_GUIDES = [
    "Uf, esas cosas pasan. 🙈 Nada grave, entras de nuevo así:\n\n"
    "📺 En la app (celular/TV):\n1. Abre la app Jellyfin\n2. Servidor: example.org\n"
    "3. Tu usuario y tu contraseña de siempre\n\n"
    "💻 Desde navegador: entra a https://example.org y listo.",
    "¡Va, tranquila! 🙌 Para entrar:\n\n📺 Si es en la app (tele o celular):\n"
    "1. Abre la app Jellyfin\n2. Donde dice servidor, escribe: example.org\n"
    "3. Pon tu usuario y contraseña de siempre",
    "1. Busca la app Jellyfin en tu celu. 2. Ábrela. 3. Te va a pedir "
    '"Dirección del servidor". Ahí escribe: 👉 example.org … Si te marca error '
    "(por ejemplo que no encuentra el servidor), mándame captura.",
    "Pa' la próxima, pídela directo en https://request.example.org — es más "
    "rápido, la pides y te llega solita.",
]

# Real leaks from the same class that must stay held.
REAL_LEAKS = [
    # operator action + HTTP status
    "Jellyfin refresh triggered successfully (204). Task is fully complete and verified.",
    # data report (carries a clock time that must not matter)
    "Revisé los logs del servidor Jellyfin (Playback Reporting) para Alex. "
    "Última vez que entró: 24 de marzo, terminando ~12:10 PM CST. Sigue pagando "
    "la cuota pero no entra desde hace 6 meses.",
    # diagnostic with the server version
    "Tu app está bien — acabo de conectarme y contesta bien (Jellyfin 10.11.11).",
    # first-person "our server" framing
    "Nuestro servidor Jellyfin está saturado ahorita.",
    # ops tool next to deployment context
    "Ya estaba registrada en el servidor (Radarr) pero sin archivo.",
    # internal addresses, even inside an otherwise-customer-shaped guide
    "Instala la app Jellyfin, conéctate a http://10.0.0.5:8096 con tu usuario.",
    "En Dirección del servidor pon 100.64.0.11:8096 y conéctate con tu usuario.",
]


@pytest.mark.parametrize("text", LOGIN_GUIDES)
def test_customer_surface_guidance_is_delivered(text):
    assert reason(text) is None


@pytest.mark.parametrize("text", REAL_LEAKS)
def test_operator_state_and_internals_stay_held(text):
    assert reason(text) == "backend_substance_disclosure"


def test_clock_time_alone_is_not_an_address():
    assert reason("Nos vemos a las 12:10 PM, terminando ~9:45.") is None
