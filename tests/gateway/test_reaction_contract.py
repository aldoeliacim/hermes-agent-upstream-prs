"""One declared reaction contract across the platform base and every adapter.

The rule this suite enforces: a capability that lives at the edges must be swept
by what it IS, not where it sits. ANY class deriving from ``BasePlatformAdapter``
that defines a native reaction primitive MUST declare ``supports_reactions`` so
callers gate on the contract instead of guessing from the presence of a method.
"""
import inspect

import pytest

from gateway.platforms.base import BasePlatformAdapter


class _ConcreteBase(BasePlatformAdapter):
    """Fills the ABC's abstract surface so the reaction contract can be exercised."""

    name = "concrete"

    async def connect(self, *, is_reconnect: bool = False) -> bool:  # pragma: no cover
        return True

    async def disconnect(self) -> None:  # pragma: no cover
        return None

    async def get_chat_info(self, chat_id):  # pragma: no cover
        return {}

    async def send(self, chat_id, content, reply_to=None, metadata=None):  # pragma: no cover
        return None


class _NoReactAdapter(_ConcreteBase):
    name = "noreact"


class _AddReactAdapter(_ConcreteBase):
    name = "addreact"
    supports_reactions = True

    async def add_reaction(self, chat_id, emoji, message_id):
        self.calls = (chat_id, emoji, message_id)
        return True

    async def remove_reaction(self, chat_id, message_id):
        self.removed = (chat_id, message_id)
        return {"ok": True}


class _RaisingReactAdapter(_ConcreteBase):
    name = "raise"
    supports_reactions = True

    async def add_reaction(self, chat_id, emoji, message_id):
        raise RuntimeError("platform down")


def _make(cls):
    """Instantiate without running the heavy __init__ (we only exercise contract)."""
    return cls.__new__(cls)


def test_base_default_is_false():
    """The base class declares the capability and defaults it OFF."""
    assert BasePlatformAdapter.supports_reactions is False


def test_base_defines_react_and_unreact():
    assert inspect.iscoroutinefunction(BasePlatformAdapter.react)
    assert inspect.iscoroutinefunction(BasePlatformAdapter.unreact)


@pytest.mark.asyncio
async def test_react_delegates_to_add_reaction():
    a = _make(_AddReactAdapter)
    ok = await a.react("chat1", "m1", "👍")
    assert ok is True
    # base react() calls add_reaction(chat_id, emoji, message_id)
    assert a.calls == ("chat1", "👍", "m1")


@pytest.mark.asyncio
async def test_unreact_delegates_and_coerces_truthy():
    a = _make(_AddReactAdapter)
    ok = await a.unreact("chat1", "m1", "👍")
    assert ok is True
    assert a.removed == ("chat1", "m1")


@pytest.mark.asyncio
async def test_react_is_safe_no_op_without_primitive():
    a = _make(_NoReactAdapter)
    assert await a.react("c", "m", "👍") is False
    assert await a.unreact("c", "m") is False


@pytest.mark.asyncio
async def test_react_never_raises_on_primitive_failure():
    a = _make(_RaisingReactAdapter)
    # A failing native primitive must be reported as False, never propagate.
    assert await a.react("c", "m", "👍") is False


# ── The invariant sweep: primitive present ⇒ capability declared ────────────
_REACTION_PRIMITIVES = (
    "add_reaction", "remove_reaction",
    "send_reaction", "_add_reaction", "_send_reaction", "_react", "_set_reaction",
)


def _adapter_classes():
    """All BasePlatformAdapter subclasses defined in the shipped platform modules."""
    import importlib

    modules = [
        "gateway.platforms.signal",
        "plugins.platforms.telegram.adapter",
        "plugins.platforms.slack.adapter",
        "plugins.platforms.discord.adapter",
        "plugins.platforms.matrix.adapter",
        "plugins.platforms.photon.adapter",
        "plugins.platforms.buzz.adapter",
        "plugins.platforms.feishu.adapter",
    ]
    seen = {}
    for mod_name in modules:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if (
                issubclass(obj, BasePlatformAdapter)
                and obj is not BasePlatformAdapter
                and obj.__module__ == mod_name
            ):
                seen[obj.__qualname__] = obj
    return list(seen.values())


def _defines_native_primitive(cls) -> bool:
    """True if cls (via its own MRO, not the base default) exposes a reaction op."""
    for prim in _REACTION_PRIMITIVES:
        member = getattr(cls, prim, None)
        if callable(member):
            return True
    return False


def test_at_least_one_shipped_adapter_is_reaction_capable():
    """Guard against the sweep passing vacuously if imports silently fail."""
    classes = _adapter_classes()
    assert classes, "no platform adapters importable — sweep would vacuously pass"
    assert any(_defines_native_primitive(c) for c in classes), (
        "expected at least one shipped adapter with a native reaction primitive"
    )


def test_every_reaction_capable_adapter_declares_the_flag():
    classes = _adapter_classes()
    offenders = [
        cls.__qualname__
        for cls in classes
        if _defines_native_primitive(cls) and not getattr(cls, "supports_reactions", False)
    ]
    assert not offenders, (
        "adapters define a native reaction primitive but do not declare "
        f"supports_reactions: {offenders}"
    )
