import assert from 'node:assert/strict';
import { test } from 'node:test';
import { installOutboundTextPolicy } from './bridge_helpers.js';

// Baileys builds a link preview by fetching the URL in the message text: an outbound network
// request driven by message content, made inside sendMessage after any caller scope check.
// installOutboundTextPolicy forces every plain-text send preview-free.

function fakeSocket() {
  const sent = [];
  return {
    sent,
    async sendMessage(jid, content, options) {
      sent.push({ jid, content, options });
      return { key: { id: 'X' } };
    },
  };
}

test('plain text sends go out with an explicit linkPreview: null', async () => {
  const sock = fakeSocket();
  installOutboundTextPolicy(sock);
  await sock.sendMessage('a@s.whatsapp.net', { text: 'see https://example.com' }, { quoted: 1 });
  assert.equal(sock.sent.length, 1);
  // Explicit null, not undefined: an undefined field lets Baileys generate the preview.
  assert.ok('linkPreview' in sock.sent[0].content, 'linkPreview must be present');
  assert.equal(sock.sent[0].content.linkPreview, null);
  assert.equal(sock.sent[0].content.text, 'see https://example.com');
  assert.deepEqual(sock.sent[0].options, { quoted: 1 }, 'options pass through untouched');
});

test("the caller's content object is never mutated", async () => {
  const sock = fakeSocket();
  installOutboundTextPolicy(sock);
  const content = { text: 'https://example.com' };
  await sock.sendMessage('a@s.whatsapp.net', content);
  assert.equal('linkPreview' in content, false);
  assert.notEqual(sock.sent[0].content, content);
});

test('a caller-supplied preview is passed through as-is', async () => {
  const sock = fakeSocket();
  installOutboundTextPolicy(sock);
  const preview = { 'canonical-url': 'https://example.com', title: 'chosen by the caller' };
  const content = { text: 'https://example.com', linkPreview: preview };
  await sock.sendMessage('a@s.whatsapp.net', content);
  assert.equal(sock.sent[0].content, content);
  assert.equal(sock.sent[0].content.linkPreview, preview);
});

test('non-text payloads (media, reactions) are not rewritten', async () => {
  const sock = fakeSocket();
  installOutboundTextPolicy(sock);
  const image = { image: Buffer.from([1]), caption: 'https://example.com' };
  const react = { react: { text: '👍', key: { id: 'k' } } };
  await sock.sendMessage('a@s.whatsapp.net', image);
  await sock.sendMessage('a@s.whatsapp.net', react);
  assert.equal(sock.sent[0].content, image);
  assert.equal(sock.sent[1].content, react);
});

test('the send result is returned to the caller', async () => {
  const sock = fakeSocket();
  installOutboundTextPolicy(sock);
  const result = await sock.sendMessage('a@s.whatsapp.net', { text: 'hi' });
  assert.deepEqual(result, { key: { id: 'X' } });
});
