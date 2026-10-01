"""Local durability contract for the external Mnemosyne KG proposer."""

from __future__ import annotations

import importlib
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest

HERMES_HOME = Path.home() / ".hermes"
sys.path.insert(0, str(HERMES_HOME))

kg = importlib.import_module("scripts.mnemosyne_kg")
inc = importlib.import_module("scripts.mnemosyne_kg.incremental")
spend = importlib.import_module("scripts.mnemosyne_kg.spend")


def _db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> sqlite3.Connection:
    path = tmp_path / "mnemosyne.db"
    monkeypatch.setattr(kg, "DB_PATH", path)
    monkeypatch.setattr(inc, "DB_PATH", path)
    monkeypatch.setattr(inc, "BUDGET_PATH", tmp_path / "spend.db")
    conn = inc._connect()
    conn.execute(
        "CREATE TABLE working_memory (id TEXT PRIMARY KEY, session_id TEXT, "
        "timestamp TEXT, source TEXT, content TEXT)"
    )
    conn.execute(
        "CREATE TABLE triples (id INTEGER PRIMARY KEY AUTOINCREMENT, subject TEXT, "
        "predicate TEXT, object TEXT, valid_from TEXT, valid_until TEXT, "
        "source TEXT, confidence REAL)"
    )
    conn.commit()
    return conn


def _add(conn: sqlite3.Connection, content: str) -> None:
    conn.execute(
        "INSERT INTO working_memory VALUES (?,?,?,?,?)",
        (
            "r1",
            "hermes_agent:main:whatsapp:dm:test",
            "2026-01-01T00:00:00+00:00",
            "conversation",
            content,
        ),
    )
    conn.commit()


def test_model_facing_tools_hide_direct_graph_mutators() -> None:
    provider_module = importlib.import_module("hermes_memory_provider")
    provider = provider_module.MnemosyneMemoryProvider()
    provider._hermes_home = str(HERMES_HOME)
    names = {schema["name"] for schema in provider.get_tool_schemas()}

    assert "mnemosyne_triple_query" in names
    assert "mnemosyne_triple_add" not in names
    assert "mnemosyne_triple_end" not in names


def test_model_output_is_staged_and_never_written_to_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn = _db(tmp_path, monkeypatch)
    _add(conn, "Sam vive en San Martín de las Pirámides y esta frase supera el mínimo")
    conn.close()
    monkeypatch.setattr(
        inc,
        "llm_triples",
        lambda _text: ([("Sam", "vive_en", "San Martín de las Pirámides")], 12),
    )

    result = inc.run(max_windows=1, max_seconds=30, verbose=False)

    conn = sqlite3.connect(tmp_path / "mnemosyne.db")
    assert result["triples"] == 0
    assert result["candidates"] == 1
    assert conn.execute("SELECT count(*) FROM triples").fetchone()[0] == 0
    assert conn.execute(
        "SELECT subject,predicate,object,status FROM kg_extraction_candidates"
    ).fetchall() == [
        ("Sam", "vive_en", "San Martín de las Pirámides", "pending")
    ]
    assert conn.execute(
        "SELECT status,triples_written FROM kg_extraction_ledger"
    ).fetchall() == [("review_pending", 0)]


def test_budget_is_persistent_and_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(spend, "DAILY_USD", 0.01)
    path = tmp_path / "spend.db"
    messages = [{"content": "hello"}]
    amount = spend.reserve(path, messages, 320)
    spend.reserve(path, messages, 320)
    with pytest.raises(spend.BudgetExhausted):
        spend.reserve(path, messages, 320)
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT reserved FROM spend").fetchone()[0] == amount * 2


def test_openrouter_request_is_private_bounded_and_no_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(inc, "_api_key", lambda: "synthetic")
    monkeypatch.setattr(inc, "BUDGET_PATH", tmp_path / "budget.db")

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self) -> bytes:
            return json.dumps(
                {"choices": [{"finish_reason": "stop", "message": {"content": '{"kg": []}'}}]}
            ).encode()

    def fake_open(request, timeout):
        body = json.loads(request.data)
        assert request.full_url == "https://openrouter.ai/api/v1/chat/completions"
        assert body["model"] == "deepseek/deepseek-v4-flash-0731"
        assert body["reasoning"] == {"enabled": False}
        assert body["provider"]["data_collection"] == "deny"
        assert body["provider"]["zdr"] is True
        assert body["provider"]["allow_fallbacks"] is False
        return Response()

    monkeypatch.setattr(inc.urllib.request, "urlopen", fake_open)
    assert inc.llm_triples("synthetic")[0] == []


def test_dashboard_hides_closed_triples_by_default(tmp_path: Path) -> None:
    module_path = HERMES_HOME / "plugins" / "mnemosyne-dashboard" / "dashboard_core.py"
    spec = importlib.util.spec_from_file_location("local_mnemosyne_dashboard_core", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    db_path = tmp_path / "dashboard.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE triples (id INTEGER PRIMARY KEY, subject TEXT, predicate TEXT, "
            "object TEXT, valid_from TEXT, valid_until TEXT, source TEXT, "
            "confidence REAL, created_at TEXT)"
        )
        conn.execute(
            "INSERT INTO triples VALUES (1,'A','rel','B',NULL,NULL,'test',1.0,'2026-01-01')"
        )
        conn.execute(
            "INSERT INTO triples VALUES (2,'Wrong','rel','Row',NULL,'2026-01-02','test',1.0,'2026-01-01')"
        )

    store = module.DashboardStore(db_path)
    assert [row["id"] for row in store.triples()] == [1]
    assert store.stats()["counts"]["triples"] == 1
    assert len(store.graph()["edges"]) == 1


def test_dashboard_exposes_candidate_evidence_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn = _db(tmp_path, monkeypatch)
    evidence = "Sam vive en San Martín de las Pirámides."
    window = inc.Window(
        "session-dashboard",
        [("r1", "2026-09-06T00:00:00+00:00", evidence)],
    )
    assert inc.stage_candidates(
        conn,
        window,
        [("Sam", "vive_en", "San Martín de las Pirámides")],
    ) == 1
    conn.commit()
    conn.close()

    plugin = HERMES_HOME / "plugins" / "mnemosyne-dashboard"
    module_path = plugin / "dashboard_core.py"
    spec = importlib.util.spec_from_file_location("local_mnemosyne_dashboard_candidates", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    ledger = module.DashboardStore(tmp_path / "mnemosyne.db").kg_candidates(
        status="pending", limit=10
    )
    assert ledger["read_only"] is True
    assert ledger["counts"]["pending"] == 1
    assert ledger["items"][0]["evidence_text"] == evidence
    assert ledger["items"][0]["status"] == "pending"

    html = (plugin / "static" / "index.html").read_text(encoding="utf-8")
    javascript = (plugin / "static" / "app.js").read_text(encoding="utf-8")
    assert 'data-panel="graphCandidates"' in html
    assert "Proposal ledger" in html
    assert "/api/kg-candidates" in javascript


def test_acceptance_requires_exact_evidence_and_records_an_audit_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn = _db(tmp_path, monkeypatch)
    evidence = "Sam vive en San Martín de las Pirámides."
    _add(conn, evidence + " Esta frase supera el mínimo requerido.")
    window = inc.Window(
        "hermes_agent:main:whatsapp:dm:test",
        [("r1", "2026-01-01T00:00:00+00:00", evidence)],
    )
    assert inc.stage_candidates(
        conn,
        window,
        [("Sam", "vive_en", "San Martín de las Pirámides")],
        proposer="synthetic-test",
    ) == 1
    candidate_id = conn.execute("SELECT id FROM kg_extraction_candidates").fetchone()[0]

    with pytest.raises(ValueError, match="evidence quote"):
        inc.review_candidate(
            conn,
            candidate_id,
            decision="accept",
            reviewer="test-reviewer",
            rationale="The source states the durable location explicitly.",
            evidence_quote="This sentence is not in the evidence window.",
        )
    assert conn.execute("SELECT count(*) FROM triples").fetchone()[0] == 0

    result = inc.review_candidate(
        conn,
        candidate_id,
        decision="accept",
        reviewer="test-reviewer",
        rationale="The source states the durable location explicitly.",
        evidence_quote=evidence,
    )
    assert result["status"] == "accepted"
    assert conn.execute(
        "SELECT subject,predicate,object,source,confidence FROM triples"
    ).fetchall() == [
        (
            "Sam",
            "vive_en",
            "San Martín de las Pirámides",
            "reviewed_extract:hermes_agent:main:whatsapp:dm:test",
            1.0,
        )
    ]
    assert conn.execute(
        "SELECT status,reviewer,review_rationale,approved_triple_id "
        "FROM kg_extraction_candidates"
    ).fetchone() == (
        "accepted",
        "test-reviewer",
        "The source states the durable location explicitly.",
        result["triple_id"],
    )
    assert conn.execute(
        "SELECT triple_id,candidate_id,verdict,evidence,rationale,reviewer "
        "FROM kg_triple_audit_events"
    ).fetchone() == (
        result["triple_id"],
        candidate_id,
        "KEEP",
        evidence,
        "The source states the durable location explicitly.",
        "test-reviewer",
    )


def test_rejected_candidate_never_creates_a_graph_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn = _db(tmp_path, monkeypatch)
    evidence = "A model guessed that a domain lives in Mexico, but the source did not say that."
    _add(conn, evidence)
    window = inc.Window(
        "hermes_agent:main:whatsapp:dm:test",
        [("r1", "2026-01-01T00:00:00+00:00", evidence)],
    )
    inc.stage_candidates(
        conn,
        window,
        [("example.com", "vive_en", "Mexico")],
        proposer="synthetic-test",
    )
    candidate_id = conn.execute("SELECT id FROM kg_extraction_candidates").fetchone()[0]

    result = inc.review_candidate(
        conn,
        candidate_id,
        decision="reject",
        reviewer="test-reviewer",
        rationale="The evidence contains the model guess, not a supported world fact.",
        evidence_quote=evidence,
    )

    assert result == {"candidate_id": candidate_id, "status": "rejected", "triple_id": None}
    assert conn.execute("SELECT count(*) FROM triples").fetchone()[0] == 0
    assert conn.execute(
        "SELECT candidate_id,verdict FROM kg_triple_audit_events"
    ).fetchone() == (candidate_id, "DROP")


def test_review_candidate_can_participate_in_a_caller_owned_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn = _db(tmp_path, monkeypatch)
    evidence = "Robin vive en Colombia."
    _add(conn, evidence)
    window = inc.Window(
        "session-atomic",
        [("r1", "2026-09-06T00:00:00+00:00", evidence)],
    )
    assert inc.stage_candidates(conn, window, [("Robin", "vive_en", "Colombia")]) == 1
    conn.commit()
    candidate_id = conn.execute("SELECT id FROM kg_extraction_candidates").fetchone()[0]

    inc.review_candidate(
        conn,
        candidate_id,
        decision="accept",
        reviewer="HAL",
        rationale="The exact sentence explicitly states the proposed relationship.",
        evidence_quote=evidence,
        commit=False,
    )
    conn.rollback()

    assert conn.execute("SELECT status FROM kg_extraction_candidates").fetchone()[0] == "pending"
    assert conn.execute("SELECT count(*) FROM triples").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM kg_triple_audit_events").fetchone()[0] == 0


def test_deterministic_backfill_stages_candidates_instead_of_writing_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn = _db(tmp_path, monkeypatch)
    conn.execute(
        "INSERT INTO working_memory VALUES (?,?,?,?,?)",
        (
            "identity-1",
            "hermes_agent:main:whatsapp:dm:contact",
            "2026-01-01T00:00:00+00:00",
            "identity",
            "Alex es hermana de Sam.",
        ),
    )
    conn.commit()
    conn.close()

    assert kg.backfill() == 1
    with sqlite3.connect(tmp_path / "mnemosyne.db") as check:
        assert check.execute("SELECT count(*) FROM triples").fetchone()[0] == 0
        assert check.execute(
            "SELECT subject,predicate,object,status,model FROM kg_extraction_candidates"
        ).fetchall() == [
            ("Alex", "es_hermana_de", "Sam", "pending", "deterministic-rules/v1")
        ]


def test_legacy_extract_session_stages_model_output_instead_of_writing_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn = _db(tmp_path, monkeypatch)
    _add(conn, "Sam vive en San Martín de las Pirámides y esta frase supera el mínimo")
    conn.close()
    monkeypatch.setattr(
        kg,
        "llm_extract",
        lambda _text: [("Sam", "vive_en", "San Martín de las Pirámides")],
    )

    assert kg.extract_session("hermes_agent:main:whatsapp:dm:test") == 1
    with sqlite3.connect(tmp_path / "mnemosyne.db") as check:
        assert check.execute("SELECT count(*) FROM triples").fetchone()[0] == 0
        assert check.execute(
            "SELECT status,model FROM kg_extraction_candidates"
        ).fetchone() == ("pending", "legacy-local-qwen/proposal-only")


@pytest.mark.parametrize(
    ("triple", "reason"),
    [
        (("Alex", "parentesco_de", "hermana"), "a family role is not the related person"),
        (("Robin", "whatsapp_de", "80000000000004@lid"), "a LID is not a phone number"),
        (("Morgan", "jid_de", "80000000000002@lid"), "a LID is not a JID"),
        (("mediahost", "cambio_en", "ZFS"), "event predicates require ISO dates"),
        (("whatsapp-bridge", "corre_en", "port 18792"), "ports require escucha_en/puerto_de"),
        (("chill-studio", "puerto_de", "70000"), "ports must be in range"),
        (("file", "path_de", "not an absolute path"), "paths must be absolute"),
        (("FuerzaFiel", "es_vm", "micro VM"), "es_vm is boolean"),
    ],
)
def test_structural_admission_rejects_typed_category_errors(
    triple: tuple[str, str, str], reason: str
) -> None:
    assert reason
    assert inc.accept_triple(*triple) is None


@pytest.mark.parametrize(
    "triple",
    [
        ("Robin", "lid_de", "80000000000004@lid"),
        ("chill-studio", "puerto_de", "8090"),
        ("chill-portal", "escucha_en", ":8091"),
        ("config.yaml", "path_de", "/home/hal/.hermes/config.yaml"),
        ("cisterna.yaml", "path_de", "C:\\Users\\owner\\esphome\\"),
        ("pve", "obsoleto_desde", "2026-08-24"),
        ("FuerzaFiel", "es_vm", "sí"),
        ("mediahost", "filesystem_de", "ZFS"),
        ("mediahost", "init_de", "systemd"),
        ("Robin", "tiene_acento", "acento paisa"),
        ("media.example.org", "dns_mode_de", "A record DNS-only"),
    ],
)
def test_structural_admission_accepts_typed_values(
    triple: tuple[str, str, str]
) -> None:
    assert inc.accept_triple(*triple) == triple
