"""A gateway-child scope must be able to die alone, promptly.

Incident (2026-08-08, second of its class): Kanban workers inherited the
gateway's cgroup, so three QA workers drove ``hermes-gateway.service`` to its
memory ceiling. Crucially the gateway did not get OOM-killed — with swap left
available the kernel chose direct-reclaim thrashing, freezing the event loop
for 20 minutes while systemd still reported the unit ``active``. An external
watchdog had to restart it.

``MemoryMax`` alone does not produce that outcome; it is ``MemorySwapMax=0``
that turns "over budget" into a prompt death instead of a host-wide grind.
The two properties are one mechanism and are asserted together: a scope that
accounts and caps memory but still has swap to fall into re-opens the exact
incident this isolation exists to prevent.

These assert the argv CONTRACT (what constraints the scope is created with),
which is the only thing observable without running systemd in the suite.
"""

from tools.process_registry import _systemd_scope_argv


def _properties(argv):
    """Collect ``--property K=V`` pairs into a dict."""
    return dict(
        argv[i + 1].split("=", 1)
        for i, token in enumerate(argv)
        if token == "--property"
    )


def test_scope_caps_memory_and_forbids_swap_fallback():
    props = _properties(_systemd_scope_argv("/usr/bin/systemd-run", "hermes-worker-x", "cmd"))

    assert props.get("MemoryAccounting") == "yes"
    assert "MemoryMax" in props, "a worker scope without a ceiling cannot protect the gateway"
    assert props.get("MemorySwapMax") == "0", (
        "MemoryMax without MemorySwapMax=0 lets an over-budget worker thrash swap instead of "
        "dying: the gateway freezes while still reporting healthy"
    )


def test_memory_ceiling_is_a_finite_positive_bound():
    props = _properties(_systemd_scope_argv("/usr/bin/systemd-run", "hermes-worker-x", "cmd"))

    assert int(props["MemoryMax"]) > 0


def test_command_is_passed_through_after_the_separator():
    argv = _systemd_scope_argv("/usr/bin/systemd-run", "hermes-worker-x", "hermes", "chat", "-q")

    assert argv[argv.index("--") + 1:] == ["hermes", "chat", "-q"]
    assert "--scope" in argv and "--user" in argv
