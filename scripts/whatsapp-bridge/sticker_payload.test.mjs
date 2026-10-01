import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mediaPayloadForFile } from './bridge_helpers.js';

// A sticker directive must produce a Baileys sticker payload, not null.
test('mediaPayloadForFile builds a sticker payload for mediaType=sticker', () => {
  const buffer = Buffer.from([0x52, 0x49, 0x46, 0x46]); // "RIFF"
  const payload = mediaPayloadForFile({ buffer, filePath: 'cat.webp', mediaType: 'sticker' });
  assert.ok(payload, 'expected a non-null payload for sticker');
  assert.equal(payload.sticker, buffer, 'sticker payload carries the raw buffer');
  // Stickers carry no caption bubble in WhatsApp.
  assert.equal(payload.caption, undefined);
});

test('mediaPayloadForFile sticker works regardless of extension', () => {
  const buffer = Buffer.from([1, 2, 3]);
  const payload = mediaPayloadForFile({ buffer, filePath: 'cat.png', mediaType: 'sticker' });
  assert.ok(payload);
  assert.equal(payload.sticker, buffer);
});
