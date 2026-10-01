"""Bridge helper-integrity hashing must cover bridge.js AND its helper modules.

Hashing only ``bridge.js`` (the old behavior) meant an edit to a helper module
never invalidated the running bridge, so a stale process kept serving old
helper code after a redeploy. The Python ``_bridge_source_hash`` and the JS
``scriptHash`` reported at ``/health`` must agree byte-for-byte, otherwise the
reuse check would loop-restart the bridge forever.
"""
import hashlib
import subprocess
from pathlib import Path

import plugins.platforms.whatsapp.adapter as wa

BRIDGE_DIR = Path(wa.__file__).resolve().parents[3] / "scripts" / "whatsapp-bridge"
BRIDGE_JS = BRIDGE_DIR / "bridge.js"


def test_source_files_cover_helpers():
    for name in ("bridge.js", "bridge_helpers.js", "allowlist.js",
                 "outbound_ids.js", "owner_message_gate.js"):
        assert name in wa._BRIDGE_SOURCE_FILES


def test_bridge_source_hash_changes_when_helper_changes(tmp_path):
    for name in wa._BRIDGE_SOURCE_FILES:
        (tmp_path / name).write_text(f"// {name}\n")
    before = wa._bridge_source_hash(tmp_path / "bridge.js")
    (tmp_path / "bridge_helpers.js").write_text("// changed\n")
    after = wa._bridge_source_hash(tmp_path / "bridge.js")
    assert before != after
    assert len(before) == 16


def test_bridge_source_hash_skips_missing(tmp_path):
    (tmp_path / "bridge.js").write_text("// only one\n")
    h = wa._bridge_source_hash(tmp_path / "bridge.js")
    assert len(h) == 16


def test_python_and_js_source_hash_agree_on_real_bridge():
    """The single most important invariant: cross-language byte-for-byte match."""
    py_hash = wa._bridge_source_hash(BRIDGE_JS)
    node_script = (
        "import { bridgeSourceHash, BRIDGE_SOURCE_FILES } "
        "from './bridge_helpers.js';"
        "process.stdout.write(bridgeSourceHash(process.cwd(), BRIDGE_SOURCE_FILES));"
    )
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", node_script],
        cwd=str(BRIDGE_DIR), capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    js_hash = proc.stdout.strip()
    assert js_hash == py_hash, f"JS {js_hash} != PY {py_hash}"
    assert len(py_hash) == 16
