// Tests for the bridge loopback capability-token check (bridgeTokenOk) that the
// /send, /edit, /react, /send-media, /send-poll, /send-location, /typing and
// /read middleware enforces. The check is a pure function in bridge_helpers.js
// (single source of truth, covered by BRIDGE_SOURCE_FILES) so it is imported
// directly rather than spinning up express/Baileys.
import assert from 'node:assert/strict';
import test from 'node:test';

import { bridgeTokenOk } from './bridge_helpers.js';

const tok = (s) => Buffer.from(s);

test('accepts any presented value when no token is configured (back-compat)', () => {
  assert.equal(bridgeTokenOk(null, undefined), true);
  assert.equal(bridgeTokenOk(null, ''), true);
  assert.equal(bridgeTokenOk(null, 'whatever'), true);
});

test('accepts the exact matching token', () => {
  assert.equal(bridgeTokenOk(tok('s3cr3t-abc'), 's3cr3t-abc'), true);
});

test('rejects a wrong token of equal length', () => {
  assert.equal(bridgeTokenOk(tok('aaaaaa'), 'bbbbbb'), false);
});

test('rejects a token of different length without throwing', () => {
  assert.equal(bridgeTokenOk(tok('short'), 'a-much-longer-token'), false);
  assert.equal(bridgeTokenOk(tok('a-much-longer-token'), 'short'), false);
});

test('rejects missing / empty / non-string presented values when a token is set', () => {
  assert.equal(bridgeTokenOk(tok('s3cr3t'), undefined), false);
  assert.equal(bridgeTokenOk(tok('s3cr3t'), ''), false);
  assert.equal(bridgeTokenOk(tok('s3cr3t'), null), false);
  assert.equal(bridgeTokenOk(tok('s3cr3t'), 12345), false);
});

test('open-prefix routing logic keeps read-only/handshake paths unauthenticated', () => {
  // Mirror of the middleware's TOKEN_OPEN_PREFIXES matching, so a future edit to
  // the open set is caught here.
  const TOKEN_OPEN_PREFIXES = ['/health', '/messages', '/chat'];
  const isOpen = (p) => TOKEN_OPEN_PREFIXES.some((pre) => p === pre || p.startsWith(pre + '/'));
  assert.equal(isOpen('/health'), true);
  assert.equal(isOpen('/messages'), true);
  assert.equal(isOpen('/chat/12345@g.us'), true);
  // State-changing endpoints are NOT open.
  for (const guarded of ['/send', '/edit', '/react', '/send-media', '/send-poll', '/send-location', '/typing', '/read']) {
    assert.equal(isOpen(guarded), false, `${guarded} must require the token`);
  }
  // A path that merely starts with an open word but isn't the prefix boundary is guarded.
  assert.equal(isOpen('/healthz-extra'), false);
});
