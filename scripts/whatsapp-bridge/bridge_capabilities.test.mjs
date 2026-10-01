import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  getStatusAuthor,
  buildDeleteKey,
  buildStarPayload,
  buildArchivePayload,
  buildMutePayload,
  buildPinPayload,
  buildMarkReadPayload,
  isValidGroupSetting,
  isValidGroupParticipantAction,
  normalizePhoneForCheck,
  isValidPrivacyUpdate,
  privacySockMethodName,
  isValidBlockAction,
  buildContactPayload,
  isValidEphemeralSeconds,
  isValidJoinApprovalMode,
  isValidMemberAddMode,
  isValidJoinRequestAction,
  isValidPresenceType,
  buildDisableLinkPreviewsPayload,
  normalizeChatJid,
} from './bridge_helpers.js';

// --- getStatusAuthor ---

test('getStatusAuthor prefers explicit fallback over cached key', () => {
  const statusMsg = { key: { participant: '5215551000001:5@s.whatsapp.net' } };
  assert.equal(getStatusAuthor(statusMsg, '5215559999999@s.whatsapp.net'), '5215559999999@s.whatsapp.net');
});

test('getStatusAuthor falls back to the cached key participant', () => {
  // The candidate's normalizeWhatsAppId strips the device suffix (`:5`) so the
  // bot's device-qualified ids compare equal to bare inbound participant ids —
  // this fixed @mention / reply-to-bot gating in groups (see bridge_helpers.js).
  // getStatusAuthor therefore returns the device-stripped author.
  const statusMsg = { key: { participant: '5215551000001:5@s.whatsapp.net' } };
  assert.equal(getStatusAuthor(statusMsg, undefined), '5215551000001@s.whatsapp.net');
});

test('getStatusAuthor falls back to a plain (no device suffix) participant', () => {
  const statusMsg = { key: { participant: '5215551000001@s.whatsapp.net' } };
  assert.equal(getStatusAuthor(statusMsg, undefined), '5215551000001@s.whatsapp.net');
});

test('getStatusAuthor returns empty string when nothing resolves', () => {
  assert.equal(getStatusAuthor({}, undefined), '');
});

// --- buildDeleteKey ---

test('buildDeleteKey requires explicit boolean fromMe (refuses to guess)', () => {
  assert.equal(buildDeleteKey({ chatId: 'c@s.whatsapp.net', messageId: 'm1', fromMe: undefined }), null);
});

test('buildDeleteKey builds the WAMessageKey for own message delete', () => {
  const key = buildDeleteKey({ chatId: 'c@s.whatsapp.net', messageId: 'm1', fromMe: true });
  assert.deepEqual(key, { remoteJid: 'c@s.whatsapp.net', id: 'm1', fromMe: true });
});

test('buildDeleteKey includes participant for group admin-delete', () => {
  const key = buildDeleteKey({
    chatId: 'g@g.us', messageId: 'm1', fromMe: false, participant: 'p@s.whatsapp.net',
  });
  assert.deepEqual(key, { remoteJid: 'g@g.us', id: 'm1', fromMe: false, participant: 'p@s.whatsapp.net' });
});

test('buildDeleteKey rejects missing chatId/messageId', () => {
  assert.equal(buildDeleteKey({ chatId: '', messageId: 'm1', fromMe: true }), null);
  assert.equal(buildDeleteKey({ chatId: 'c@s.whatsapp.net', messageId: '', fromMe: true }), null);
});

// --- buildStarPayload ---

test('buildStarPayload builds a chatModify star payload', () => {
  const payload = buildStarPayload({ messages: [{ id: 'm1', fromMe: true }], star: true });
  assert.deepEqual(payload, { star: { messages: [{ id: 'm1', fromMe: true }], star: true } });
});

test('buildStarPayload drops malformed entries and returns null if none remain', () => {
  assert.equal(buildStarPayload({ messages: [{ id: 'm1' }], star: true }), null);
  assert.equal(buildStarPayload({ messages: [], star: true }), null);
  assert.equal(buildStarPayload({ messages: null, star: true }), null);
});

test('buildStarPayload filters partial-bad entries out of a mixed list', () => {
  const payload = buildStarPayload({
    messages: [{ id: 'm1', fromMe: true }, { id: 'bad' }],
    star: false,
  });
  assert.deepEqual(payload, { star: { messages: [{ id: 'm1', fromMe: true }], star: false } });
});

// --- buildArchivePayload / buildMutePayload / buildPinPayload / buildMarkReadPayload ---

test('buildArchivePayload requires boolean archive + a lastMessage with a key', () => {
  assert.equal(buildArchivePayload({ archive: true, lastMessage: null }), null);
  assert.equal(buildArchivePayload({ archive: 'yes', lastMessage: { key: {} } }), null);
  assert.deepEqual(
    buildArchivePayload({ archive: true, lastMessage: { key: { id: 'm1' } } }),
    { archive: true, lastMessages: [{ key: { id: 'm1' } }] },
  );
});

test('buildMutePayload accepts a positive duration or null (unmute)', () => {
  assert.deepEqual(buildMutePayload({ muteMs: 28800000 }), { mute: 28800000 });
  assert.deepEqual(buildMutePayload({ muteMs: null }), { mute: null });
  assert.equal(buildMutePayload({ muteMs: -1 }), null);
  assert.equal(buildMutePayload({ muteMs: NaN }), null);
});

test('buildPinPayload requires a boolean', () => {
  assert.deepEqual(buildPinPayload({ pin: true }), { pin: true });
  assert.equal(buildPinPayload({ pin: undefined }), null);
});

test('buildMarkReadPayload requires boolean read + a lastMessage with a key', () => {
  assert.deepEqual(
    buildMarkReadPayload({ read: false, lastMessage: { key: { id: 'm1' } } }),
    { markRead: false, lastMessages: [{ key: { id: 'm1' } }] },
  );
  assert.equal(buildMarkReadPayload({ read: false, lastMessage: null }), null);
});

// --- group setting / participant action validators ---

test('isValidGroupSetting accepts only the four WA-recognized tags', () => {
  assert.equal(isValidGroupSetting('announcement'), true);
  assert.equal(isValidGroupSetting('not_announcement'), true);
  assert.equal(isValidGroupSetting('locked'), true);
  assert.equal(isValidGroupSetting('unlocked'), true);
  assert.equal(isValidGroupSetting('delete_group'), false);
  assert.equal(isValidGroupSetting(''), false);
});

test('isValidGroupParticipantAction accepts only add/remove/promote/demote', () => {
  assert.equal(isValidGroupParticipantAction('add'), true);
  assert.equal(isValidGroupParticipantAction('remove'), true);
  assert.equal(isValidGroupParticipantAction('promote'), true);
  assert.equal(isValidGroupParticipantAction('demote'), true);
  assert.equal(isValidGroupParticipantAction('nuke'), false);
});

// --- normalizePhoneForCheck ---

test('normalizePhoneForCheck strips + and builds an s.whatsapp.net JID', () => {
  assert.equal(normalizePhoneForCheck('+5215551000001'), '5215551000001@s.whatsapp.net');
  assert.equal(normalizePhoneForCheck('5215551000001'), '5215551000001@s.whatsapp.net');
});

test('normalizePhoneForCheck passes through an already-built s.whatsapp.net JID', () => {
  assert.equal(normalizePhoneForCheck('5215551000001@s.whatsapp.net'), '5215551000001@s.whatsapp.net');
});

test('normalizePhoneForCheck rejects LIDs (onWhatsApp does not support them)', () => {
  assert.equal(normalizePhoneForCheck('140000000000003@lid'), null);
});

test('normalizePhoneForCheck rejects empty input', () => {
  assert.equal(normalizePhoneForCheck(''), null);
  assert.equal(normalizePhoneForCheck(undefined), null);
});

// --- Tier 2: privacy, blocklist, contacts, group-join-approval, ephemeral,
// presence ---

test('isValidPrivacyUpdate accepts each category\'s real WhatsApp enum', () => {
  assert.equal(isValidPrivacyUpdate('last', 'contacts'), true);
  assert.equal(isValidPrivacyUpdate('online', 'match_last_seen'), true);
  assert.equal(isValidPrivacyUpdate('profile', 'none'), true);
  assert.equal(isValidPrivacyUpdate('status', 'contact_blacklist'), true);
  assert.equal(isValidPrivacyUpdate('readreceipts', 'all'), true);
  assert.equal(isValidPrivacyUpdate('groupadd', 'contacts'), true);
  assert.equal(isValidPrivacyUpdate('calladd', 'known'), true);
  assert.equal(isValidPrivacyUpdate('messages', 'contacts'), true);
});

test('isValidPrivacyUpdate rejects a value from the WRONG category\'s enum', () => {
  // 'online' only accepts all/match_last_seen — 'none' is valid for other
  // categories but not this one. Category-specific validation, not a single
  // shared enum, is the whole point.
  assert.equal(isValidPrivacyUpdate('online', 'none'), false);
  assert.equal(isValidPrivacyUpdate('groupadd', 'none'), false);
  assert.equal(isValidPrivacyUpdate('messages', 'contact_blacklist'), false);
});

test('isValidPrivacyUpdate rejects an unknown category outright', () => {
  assert.equal(isValidPrivacyUpdate('nonexistent', 'all'), false);
});

test('privacySockMethodName maps every category to its real sock method', () => {
  assert.equal(privacySockMethodName('last'), 'updateLastSeenPrivacy');
  assert.equal(privacySockMethodName('online'), 'updateOnlinePrivacy');
  assert.equal(privacySockMethodName('profile'), 'updateProfilePicturePrivacy');
  assert.equal(privacySockMethodName('status'), 'updateStatusPrivacy');
  assert.equal(privacySockMethodName('readreceipts'), 'updateReadReceiptsPrivacy');
  assert.equal(privacySockMethodName('groupadd'), 'updateGroupsAddPrivacy');
  assert.equal(privacySockMethodName('calladd'), 'updateCallPrivacy');
  assert.equal(privacySockMethodName('messages'), 'updateMessagesPrivacy');
  assert.equal(privacySockMethodName('bogus'), null);
});

test('isValidBlockAction accepts only block/unblock', () => {
  assert.equal(isValidBlockAction('block'), true);
  assert.equal(isValidBlockAction('unblock'), true);
  assert.equal(isValidBlockAction('ban'), false);
});

test('buildContactPayload builds an add-contact chatModify payload', () => {
  assert.deepEqual(buildContactPayload({ name: 'Robin' }), { contact: { name: 'Robin' } });
});

test('buildContactPayload builds a remove-contact payload regardless of name', () => {
  assert.deepEqual(buildContactPayload({ name: 'Robin', remove: true }), { contact: null });
  assert.deepEqual(buildContactPayload({ remove: true }), { contact: null });
});

test('buildContactPayload rejects an empty add (use remove instead)', () => {
  assert.equal(buildContactPayload({ name: '' }), null);
  assert.equal(buildContactPayload({ name: '   ' }), null);
});

test('isValidEphemeralSeconds accepts only WhatsApp\'s four durations', () => {
  assert.equal(isValidEphemeralSeconds(0), true);
  assert.equal(isValidEphemeralSeconds(86400), true);
  assert.equal(isValidEphemeralSeconds(604800), true);
  assert.equal(isValidEphemeralSeconds(7776000), true);
  assert.equal(isValidEphemeralSeconds(3600), false);
  assert.equal(isValidEphemeralSeconds('86400'), true); // Number() coercion
});

test('isValidJoinApprovalMode accepts only on/off', () => {
  assert.equal(isValidJoinApprovalMode('on'), true);
  assert.equal(isValidJoinApprovalMode('off'), true);
  assert.equal(isValidJoinApprovalMode('maybe'), false);
});

test('isValidMemberAddMode accepts only all_member_add/admin_add', () => {
  assert.equal(isValidMemberAddMode('all_member_add'), true);
  assert.equal(isValidMemberAddMode('admin_add'), true);
  assert.equal(isValidMemberAddMode('nobody'), false);
});

test('isValidJoinRequestAction accepts only approve/reject', () => {
  assert.equal(isValidJoinRequestAction('approve'), true);
  assert.equal(isValidJoinRequestAction('reject'), true);
  assert.equal(isValidJoinRequestAction('ignore'), false);
});

test('isValidPresenceType accepts the five Baileys presence types', () => {
  assert.equal(isValidPresenceType('available'), true);
  assert.equal(isValidPresenceType('unavailable'), true);
  assert.equal(isValidPresenceType('composing'), true);
  assert.equal(isValidPresenceType('recording'), true);
  assert.equal(isValidPresenceType('paused'), true);
  assert.equal(isValidPresenceType('typing'), false);
});

// --- Tier 4: link-preview privacy ---

test('buildDisableLinkPreviewsPayload requires an explicit boolean', () => {
  assert.deepEqual(
    buildDisableLinkPreviewsPayload({ disabled: true }),
    { disableLinkPreviews: { isPreviewsDisabled: true } },
  );
  assert.deepEqual(
    buildDisableLinkPreviewsPayload({ disabled: false }),
    { disableLinkPreviews: { isPreviewsDisabled: false } },
  );
  assert.equal(buildDisableLinkPreviewsPayload({ disabled: undefined }), null);
  assert.equal(buildDisableLinkPreviewsPayload({}), null);
});

// --- GET /me chaining fix: normalizeChatJid must strip the device suffix
// so sock.user.id/lid can feed straight into USync-based lookups
// (fetchStatus/fetchDisappearingDuration), which silently no-op (return
// {results: []}, not an error) on a device-suffixed JID. Confirmed live
// 2026-08-11: a raw own JID with a device suffix (e.g.
// "555.../0000:23@s.whatsapp.net") returns {results: []}; the same JID
// with the suffix stripped returns the real status text. ---

test('normalizeChatJid strips a device suffix from an own-account JID', () => {
  assert.equal(
    normalizeChatJid('5215551000001:23@s.whatsapp.net'),
    '5215551000001@s.whatsapp.net',
  );
  assert.equal(
    normalizeChatJid('9987654321000:23@lid'),
    '9987654321000@lid',
  );
});

test('normalizeChatJid is a no-op on an already-bare JID', () => {
  assert.equal(
    normalizeChatJid('5215551000001@s.whatsapp.net'),
    '5215551000001@s.whatsapp.net',
  );
});

test('normalizeChatJid preserves the server domain (lid vs s.whatsapp.net never collapse)', () => {
  assert.notEqual(
    normalizeChatJid('9987654321000:23@lid'),
    normalizeChatJid('9987654321000:23@s.whatsapp.net'),
  );
});
