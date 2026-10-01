"""The WhatsApp platform registers a factory *callable* (``_build_adapter``),
not the class directly, so adapter construction has a single seam that
registration wiring and future config normalization can hook without touching
``ctx.register_platform`` call sites.
"""
from unittest.mock import MagicMock, patch

import plugins.platforms.whatsapp.adapter as wa


def test_build_adapter_delegates_to_whatsapp_adapter():
    sentinel_config = object()
    with patch.object(wa, "WhatsAppAdapter") as MockAdapter:
        MockAdapter.return_value = "built-adapter"
        result = wa._build_adapter(sentinel_config)
    MockAdapter.assert_called_once_with(sentinel_config)
    assert result == "built-adapter"


def test_register_wires_build_adapter_factory():
    ctx = MagicMock()
    wa.register(ctx)
    ctx.register_platform.assert_called_once()
    kwargs = ctx.register_platform.call_args.kwargs
    # The factory is the module-level callable, not the class object.
    assert kwargs["adapter_factory"] is wa._build_adapter
    assert kwargs["adapter_factory"] is not wa.WhatsAppAdapter
