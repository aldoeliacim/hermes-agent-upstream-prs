import test from 'node:test';
import assert from 'node:assert/strict';

import { classifyOwnerMessageGate } from './owner_message_gate.js';

function makeRecentlySent(ids = []) {
  const set = new Set(ids);
  return { has: (id) => set.has(id) };
}

function makeAllowlist(allowedChatIds) {
  if (allowedChatIds === '*') {
    return () => true;
  }
  const set = new Set(allowedChatIds);
  return (id) => set.has(id);
}

test('non-fromMe messages always pass through', () => {
  const decision = classifyOwnerMessageGate({
    fromMe: false,
    fromOwnerEnabled: true,
    recentlySent: makeRecentlySent(),
    allowlistMatches: makeAllowlist([]),
    messageId: 'M1',
    chatId: '6281234567890@s.whatsapp.net',
  });
  assert.deepEqual(decision, { action: 'pass' });
});

test('fromMe echo of our own /send is dropped', () => {
  const decision = classifyOwnerMessageGate({
    fromMe: true,
    fromOwnerEnabled: true,
    recentlySent: makeRecentlySent(['M-OWN-1']),
    allowlistMatches: makeAllowlist('*'),
    messageId: 'M-OWN-1',
    chatId: '6281234567890@s.whatsapp.net',
  });
  assert.deepEqual(decision, { action: 'drop_echo' });
});

test('fromMe is dropped when forwarding is disabled', () => {
  const decision = classifyOwnerMessageGate({
    fromMe: true,
    fromOwnerEnabled: false,
    recentlySent: makeRecentlySent(),
    allowlistMatches: makeAllowlist('*'),
    messageId: 'M-OWN-2',
    chatId: '6281234567890@s.whatsapp.net',
  });
  assert.deepEqual(decision, { action: 'drop_disabled' });
});

test('fromMe is dropped when chatId is not on the allowlist (regression)', () => {
  // This is the bug. Before the fix, an owner reply in a non-allowlisted
  // chat was still forwarded with fromOwner: true, which made the
  // gateway-policy owner-implicit branch create stray handover rows for
  // the non-allowlisted contact.
  const decision = classifyOwnerMessageGate({
    fromMe: true,
    fromOwnerEnabled: true,
    recentlySent: makeRecentlySent(),
    allowlistMatches: makeAllowlist(['6281234567890@s.whatsapp.net']),
    messageId: 'M-OWN-3',
    chatId: '111600547700784@lid',
  });
  assert.deepEqual(decision, { action: 'drop_allowlist' });
});

test('fromMe is forwarded as owner when chatId is allowlisted', () => {
  const decision = classifyOwnerMessageGate({
    fromMe: true,
    fromOwnerEnabled: true,
    recentlySent: makeRecentlySent(),
    allowlistMatches: makeAllowlist(['6281234567890@s.whatsapp.net']),
    messageId: 'M-OWN-4',
    chatId: '6281234567890@s.whatsapp.net',
  });
  assert.deepEqual(decision, { action: 'forward_owner' });
});

test('open-allowlist (matchesAllowedUser short-circuits true) forwards as owner', () => {
  // matchesAllowedUser returns true on empty allowlist or "*"; the gate
  // must respect that so deployments without an allowlist are unaffected
  // by the new check.
  const decision = classifyOwnerMessageGate({
    fromMe: true,
    fromOwnerEnabled: true,
    recentlySent: makeRecentlySent(),
    allowlistMatches: () => true,
    messageId: 'M-OWN-5',
    chatId: '111600547700784@lid',
  });
  assert.deepEqual(decision, { action: 'forward_owner' });
});

test('echo check fires before allowlist check', () => {
  // A bot-API echo whose chatId happens to be off-allowlist should still
  // be dropped as drop_echo, not drop_allowlist, so logging stays
  // honest about the actual reason.
  const decision = classifyOwnerMessageGate({
    fromMe: true,
    fromOwnerEnabled: true,
    recentlySent: makeRecentlySent(['M-ECHO-1']),
    allowlistMatches: makeAllowlist([]),
    messageId: 'M-ECHO-1',
    chatId: '111600547700784@lid',
  });
  assert.deepEqual(decision, { action: 'drop_echo' });
});

test('disabled flag fires before allowlist check', () => {
  // Pre-existing deployments with WHATSAPP_FORWARD_OWNER_MESSAGES unset
  // must see drop_disabled regardless of allowlist state, otherwise
  // every fromMe message would log a misleading allowlist_mismatch.
  const decision = classifyOwnerMessageGate({
    fromMe: true,
    fromOwnerEnabled: false,
    recentlySent: makeRecentlySent(),
    allowlistMatches: makeAllowlist([]),
    messageId: 'M-OWN-6',
    chatId: '111600547700784@lid',
  });
  assert.deepEqual(decision, { action: 'drop_disabled' });
});

// --- groups -----------------------------------------------------------------
// The owner's own message in an allowlisted group used to be dropped by a
// caller-side pre-filter, before this gate ran. Every other member's message in
// that same group was forwarded, so the owner's were the only traffic that never
// arrived. Groups are decided here instead, against the GROUP allowlist.

test('fromMe in an allowlisted group is forwarded without the DM opt-in', () => {
  const decision = classifyOwnerMessageGate({
    fromMe: true,
    fromOwnerEnabled: false,          // DM opt-in deliberately off
    recentlySent: new Set(),
    allowlistMatches: () => false,    // DM allowlist irrelevant for a group
    messageId: 'GROUPMSG1',
    chatId: '5215514706713-1463600871@g.us',
    isGroup: true,
    groupAllowed: true,
  });
  assert.deepEqual(decision, { action: 'forward_owner' });
});

test('fromMe in a group that is NOT allowlisted is dropped (regression)', () => {
  const decision = classifyOwnerMessageGate({
    fromMe: true,
    fromOwnerEnabled: true,
    recentlySent: new Set(),
    allowlistMatches: () => true,
    messageId: 'GROUPMSG2',
    chatId: '120363999999999999@g.us',
    isGroup: true,
    groupAllowed: false,
  });
  assert.deepEqual(decision, { action: 'drop_allowlist' });
});

test('group echo of our own /send is still dropped', () => {
  const decision = classifyOwnerMessageGate({
    fromMe: true,
    fromOwnerEnabled: false,
    recentlySent: new Set(['GROUPMSG3']),
    allowlistMatches: () => false,
    messageId: 'GROUPMSG3',
    chatId: '5215514706713-1463600871@g.us',
    isGroup: true,
    groupAllowed: true,
  });
  assert.deepEqual(decision, { action: 'drop_echo' });
});

test('non-fromMe group traffic still passes through untouched', () => {
  const decision = classifyOwnerMessageGate({
    fromMe: false,
    isGroup: true,
    groupAllowed: false,
    messageId: 'GROUPMSG4',
    chatId: '120363999999999999@g.us',
  });
  assert.deepEqual(decision, { action: 'pass' });
});
