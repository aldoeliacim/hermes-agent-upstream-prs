"""Do not attach or redirect scoped Relay bearer auth to foreign URLs."""
from aiohttp import web
import pytest
from gateway.relay.media import RelayMediaClient


@pytest.mark.asyncio
async def test_foreign_namespace_never_fetches_or_receives_bearer():
    seen = []
    async def sink(request):
        seen.append(request.path)
        return web.Response(body=b'synthetic foreign data')
    async def redirect(request):
        raise web.HTTPFound('/gateways/b/relay/media/file')
    app = web.Application()
    app.router.add_get('/gateways/a/relay/media/redirect', redirect)
    app.router.add_get('/gateways/b/relay/media/file', sink)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, '127.0.0.1', 0).start()
    try:
        origin = f'http://127.0.0.1:{runner.addresses[0][1]}'
        mc = RelayMediaClient(origin + '/gateways/a', 'a', 'synthetic-only')
        foreign = origin + '/gateways/b/relay/media/file'
        assert not mc.is_relay_media_url(foreign)
        assert await mc.download(foreign) is None
        assert seen == []
        assert await mc.download(origin + '/gateways/a/relay/media/redirect') is None
        assert seen == []
    finally:
        await runner.cleanup()


@pytest.mark.parametrize('url', [
    'https://foreign.invalid/gateways/a/relay/media/file',
    'https://relay.invalid/gateways/a/relay/media/../other',
    'https://relay.invalid/gateways/a/relay/media/%2e%2e',
    'https://relay.invalid/gateways/a/relay/media/file?secret=1',
])
def test_only_exact_namespace_gets_bearer(url):
    mc = RelayMediaClient('https://relay.invalid/gateways/a', 'a', 'synthetic-only')
    assert not mc.is_relay_media_url(url)
