/**
 * Regression test for inbound WhatsApp media CDN-expiry recovery.
 *
 * Baileys' downloadMediaMessage only retries via reuploadRequest when
 * `error.status` is 410/404. A @hapi/boom error exposes its code at
 * `error.output.statusCode` — `error.status` is undefined — and WhatsApp's CDN
 * answers an expired media URL with 403, which isn't in Baileys' retry set at
 * all. So the recovery path never fires and every expired sticker/image
 * degrades to "[media could not be downloaded]". We retry it ourselves.
 *
 * Ported from live 828ce02edf bridge.sticker.test.mjs (block 1), neutral fixtures.
 */

import { strict as assert } from 'node:assert';

import { downloadMediaWithRecovery } from './bridge_helpers.js';

// A @hapi/boom-shaped error: statusCode lives under output, NOT on .status.
const boomLike = (statusCode) => {
  const err = new Error('Failed to fetch stream from https://mmg.whatsapp.net/x.enc');
  err.isBoom = true;
  err.output = { statusCode };
  return err;
};

for (const status of [403, 410, 404]) {
  let attempts = 0;
  let reuploadCalls = 0;
  const downloaded = await downloadMediaWithRecovery({
    msg: { key: { id: 'm1' }, message: { stickerMessage: {} } },
    download: async () => {
      attempts += 1;
      if (attempts === 1) throw boomLike(status);
      return Buffer.from('sticker-bytes');
    },
    reuploadRequest: async (m) => {
      reuploadCalls += 1;
      return m;
    },
  });
  assert.equal(reuploadCalls, 1, `status ${status} must trigger exactly one reupload request`);
  assert.equal(attempts, 2, `status ${status} must be retried after the reupload`);
  assert.equal(downloaded.toString(), 'sticker-bytes');
}

// A non-recoverable failure must NOT loop: one reupload attempt, then throw.
{
  let attempts = 0;
  let reuploadCalls = 0;
  await assert.rejects(
    downloadMediaWithRecovery({
      msg: { key: { id: 'm2' }, message: { stickerMessage: {} } },
      download: async () => {
        attempts += 1;
        throw boomLike(403);
      },
      reuploadRequest: async (m) => {
        reuploadCalls += 1;
        return m;
      },
    }),
  );
  assert.equal(reuploadCalls, 1, 'must not retry the reupload more than once');
  assert.equal(attempts, 2, 'must not loop downloads forever');
}

// A non-HTTP error (bad decrypt, disk full) must surface immediately with no
// reupload attempt — reuploading cannot fix it.
{
  let plainReuploads = 0;
  await assert.rejects(
    downloadMediaWithRecovery({
      msg: { key: { id: 'm3' }, message: { stickerMessage: {} } },
      download: async () => {
        throw new Error('bad decrypt');
      },
      reuploadRequest: async (m) => {
        plainReuploads += 1;
        return m;
      },
    }),
    /bad decrypt/,
  );
  assert.equal(plainReuploads, 0, 'a non-HTTP failure must not request a reupload');
}

console.log('bridge.media_recovery.test.mjs: OK');
