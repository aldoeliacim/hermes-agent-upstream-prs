import { test } from 'node:test';
import assert from 'node:assert/strict';

import { createBoundedMessageStore } from './bridge_helpers.js';

const msg = (id, remoteJid) => ({ key: { id, remoteJid } });

test('latestForChat returns the most recent remembered message for a chat', () => {
  const store = createBoundedMessageStore(8);
  store.remember(msg('a1', 'chatA@s.whatsapp.net'));
  store.remember(msg('b1', 'chatB@s.whatsapp.net'));
  store.remember(msg('a2', 'chatA@s.whatsapp.net'));

  assert.equal(store.latestForChat('chatA@s.whatsapp.net').key.id, 'a2');
  assert.equal(store.latestForChat('chatB@s.whatsapp.net').key.id, 'b1');
});

test('latestForChat returns null for an unknown chat or falsy id', () => {
  const store = createBoundedMessageStore(8);
  store.remember(msg('a1', 'chatA@s.whatsapp.net'));

  assert.equal(store.latestForChat('nobody@s.whatsapp.net'), null);
  assert.equal(store.latestForChat(''), null);
  assert.equal(store.latestForChat(undefined), null);
});

test('latestForChat follows re-remembered ordering', () => {
  const store = createBoundedMessageStore(8);
  store.remember(msg('a1', 'chatA@s.whatsapp.net'));
  store.remember(msg('a2', 'chatA@s.whatsapp.net'));
  // Re-remembering a1 moves it to the newest position (Map delete+set).
  store.remember(msg('a1', 'chatA@s.whatsapp.net'));

  assert.equal(store.latestForChat('chatA@s.whatsapp.net').key.id, 'a1');
});

test('ttl: expired entries are gone for get, peek and latestInboundForChat', () => {
  const store = createBoundedMessageStore(8, { ttlMs: 1000 });
  const now = Date.now();
  store.remember(msg('old', 'chatA@s.whatsapp.net'), now - 5000);
  store.remember(msg('new', 'chatA@s.whatsapp.net'), now - 10);

  assert.equal(store.peek('old', now), null);
  assert.equal(store.get('old'), null);
  assert.equal(store.latestInboundForChat('chatA@s.whatsapp.net', { now }).key.id, 'new');
});

test('peek does not disturb LRU order; get does (documented contract)', () => {
  const store = createBoundedMessageStore(2);
  store.remember(msg('a', 'c@s.whatsapp.net'));
  store.remember(msg('b', 'c@s.whatsapp.net'));

  // peek(a) must NOT refresh a; remembering c should evict a (oldest).
  store.peek('a');
  store.remember(msg('c', 'c@s.whatsapp.net'));
  assert.equal(store.peek('a'), null);
  assert.equal(store.peek('b').key.id, 'b');

  // get(b) refreshes b; remembering d should evict c, not b.
  store.get('b');
  store.remember(msg('d', 'c@s.whatsapp.net'));
  assert.equal(store.peek('c'), null);
  assert.equal(store.peek('b').key.id, 'b');
});

test('latestInboundForChat skips outbound and normalizes device suffixes', () => {
  const store = createBoundedMessageStore(8);
  const now = Date.now();
  store.remember({ key: { id: 'out', remoteJid: 'chatA@s.whatsapp.net', fromMe: true } }, now - 5);
  store.remember({ key: { id: 'in', remoteJid: 'chatA:17@s.whatsapp.net', fromMe: false } }, now - 50);

  const hit = store.latestInboundForChat('chatA@s.whatsapp.net', { now });
  assert.equal(hit.key.id, 'in');
});
