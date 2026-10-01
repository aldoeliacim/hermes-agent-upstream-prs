import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { bridgeSourceHash, BRIDGE_SOURCE_FILES } from './bridge_helpers.js';

test('BRIDGE_SOURCE_FILES covers bridge.js and its helper modules', () => {
  for (const name of ['bridge.js', 'bridge_helpers.js', 'allowlist.js', 'outbound_ids.js', 'owner_message_gate.js']) {
    assert.ok(BRIDGE_SOURCE_FILES.includes(name), `${name} must be hashed`);
  }
});

test('bridgeSourceHash is a stable 16-hex-char digest', () => {
  const dir = mkdtempSync(join(tmpdir(), 'wa-hash-'));
  try {
    for (const name of BRIDGE_SOURCE_FILES) writeFileSync(join(dir, name), `// ${name}\n`);
    const a = bridgeSourceHash(dir, BRIDGE_SOURCE_FILES);
    const b = bridgeSourceHash(dir, BRIDGE_SOURCE_FILES);
    assert.match(a, /^[0-9a-f]{16}$/);
    assert.equal(a, b, 'hash is deterministic');
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('bridgeSourceHash changes when ANY helper file changes', () => {
  const dir = mkdtempSync(join(tmpdir(), 'wa-hash-'));
  try {
    for (const name of BRIDGE_SOURCE_FILES) writeFileSync(join(dir, name), `// ${name}\n`);
    const before = bridgeSourceHash(dir, BRIDGE_SOURCE_FILES);
    writeFileSync(join(dir, 'bridge_helpers.js'), '// changed\n');
    const after = bridgeSourceHash(dir, BRIDGE_SOURCE_FILES);
    assert.notEqual(before, after, 'a helper edit must invalidate the hash');
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('bridgeSourceHash is order-independent (sorts internally)', () => {
  const dir = mkdtempSync(join(tmpdir(), 'wa-hash-'));
  try {
    for (const name of BRIDGE_SOURCE_FILES) writeFileSync(join(dir, name), `// ${name}\n`);
    const a = bridgeSourceHash(dir, BRIDGE_SOURCE_FILES);
    const b = bridgeSourceHash(dir, [...BRIDGE_SOURCE_FILES].reverse());
    assert.equal(a, b);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('bridgeSourceHash skips missing files gracefully', () => {
  const dir = mkdtempSync(join(tmpdir(), 'wa-hash-'));
  try {
    writeFileSync(join(dir, 'bridge.js'), '// only one\n');
    const h = bridgeSourceHash(dir, BRIDGE_SOURCE_FILES);
    assert.match(h, /^[0-9a-f]{16}$/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
