import path from 'path';
import { mkdirSync, writeFileSync, readFileSync } from 'fs';
import { randomBytes, createHash, timingSafeEqual } from 'crypto';

/**
 * Constant-time capability-token check for the bridge's loopback HTTP endpoints.
 * `tokenBuf` is the configured token as a Buffer (or null when no token is set).
 * Returns true when no token is configured (back-compat), or when `presented`
 * matches `tokenBuf` in constant time. Length mismatch bails before
 * timingSafeEqual (which throws on unequal lengths); any throw is treated as a
 * non-match, never an accept.
 */
export function bridgeTokenOk(tokenBuf, presented) {
  if (!tokenBuf) return true;
  const got = Buffer.from(String(presented || ''));
  if (got.length !== tokenBuf.length) return false;
  try {
    return timingSafeEqual(got, tokenBuf);
  } catch {
    return false;
  }
}


// Bridge source files whose integrity the running process advertises via the
// /health `scriptHash`. Hashing bridge.js alone let a helper-module edit slip
// past the reuse check, so a stale process kept serving old helper code after a
// redeploy. The Python side (_bridge_source_hash) replicates this byte-for-byte.
export const BRIDGE_SOURCE_FILES = [
  'allowlist.js',
  'bridge.js',
  'bridge_helpers.js',
  'outbound_ids.js',
  'owner_message_gate.js',
];

/**
 * Deterministic 16-hex-char digest over the given source files in `dir`.
 * Sorts names, feeds `name\0bytes\0` per file, skips missing files. MUST match
 * gateway/plugins/platforms/whatsapp/adapter.py:_bridge_source_hash exactly.
 */
export function bridgeSourceHash(dir, files = BRIDGE_SOURCE_FILES) {
  const h = createHash('sha256');
  for (const name of [...files].sort()) {
    let data;
    try {
      data = readFileSync(path.join(dir, name));
    } catch {
      continue;
    }
    h.update(Buffer.from(name, 'utf8'));
    h.update(Buffer.from([0]));
    h.update(data);
    h.update(Buffer.from([0]));
  }
  return h.digest('hex').slice(0, 16);
}

export const MIME_MAP = {
  jpg: 'image/jpeg', jpeg: 'image/jpeg', png: 'image/png',
  webp: 'image/webp', gif: 'image/gif',
  mp4: 'video/mp4', mov: 'video/quicktime', avi: 'video/x-msvideo',
  mkv: 'video/x-matroska', '3gp': 'video/3gpp',
  pdf: 'application/pdf',
  doc: 'application/msword',
  docx: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  xlsx: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
};

export function normalizeWhatsAppId(value) {
  if (!value) return '';
  // Baileys reports the bot's own ids device-qualified (`<user>:<device>@lid`), while
  // inbound mentionedJid / contextInfo.participant are not. Drop the suffix so both
  // forms compare equal; the old `':' -> '@'` swap produced `<user>@<device>@lid`,
  // which never matched and silently broke @mention / reply-to-bot gating in groups.
  return String(value).replace(/:\d+(?=@)/, '').replace(/:\d+$/, '');
}

function unwrapMessageEnvelopes(content) {
  let cur = content;
  // Envelopes nest (ephemeral wrapping viewOnce wrapping the payload); peel
  // until an inner message is reached so nested quotes resolve too.
  for (let i = 0; i < 8 && cur; i++) {
    const next =
      cur.ephemeralMessage?.message ??
      cur.viewOnceMessage?.message ??
      cur.viewOnceMessageV2?.message ??
      cur.documentWithCaptionMessage?.message;
    if (next === undefined) break;
    cur = next;
  }
  return cur;
}

export function getMessageContent(msg) {
  const raw = msg?.message || {};
  const content = unwrapMessageEnvelopes(raw);
  // A peeled envelope returns its inner message immediately: the payload
  // shape (template/buttons/list/etc.) applies to unenveloped messages.
  if (content !== raw) return content;
  if (content.templateMessage?.hydratedTemplate) return content.templateMessage.hydratedTemplate;
  if (content.buttonsMessage) return content.buttonsMessage;
  if (content.listMessage) return content.listMessage;
  return content;
}

export function getContextInfo(messageContent) {
  if (!messageContent || typeof messageContent !== 'object') return {};
  for (const value of Object.values(messageContent)) {
    if (value && typeof value === 'object' && value.contextInfo) {
      return value.contextInfo;
    }
  }
  return {};
}

export function createBoundedMessageStore(limit = 512, { ttlMs = 0 } = {}) {
  // id -> { msg, seenAt }. Insertion order doubles as the LRU order.
  const byId = new Map();

  function expired(entry, now) {
    return ttlMs > 0 && now - entry.seenAt > ttlMs;
  }

  function evict(now) {
    if (ttlMs > 0) {
      for (const [id, entry] of byId) {
        if (expired(entry, now)) byId.delete(id);
      }
    }
    while (byId.size > limit) {
      const oldest = byId.keys().next().value;
      byId.delete(oldest);
    }
  }

  function remember(msg, seenAt = Date.now()) {
    const id = msg?.key?.id;
    if (!id) return;
    byId.delete(id);
    byId.set(id, { msg, seenAt });
    evict(seenAt);
  }

  function get(id) {
    // LRU touch: refreshes the entry's position (NOT its seenAt). Use peek()
    // for read-only resolution where recency order must stay untouched.
    if (!id || !byId.has(id)) return null;
    const entry = byId.get(id);
    if (expired(entry, Date.now())) {
      byId.delete(id);
      return null;
    }
    byId.delete(id);
    byId.set(id, entry);
    return entry.msg;
  }

  function peek(id, now = Date.now()) {
    // Non-mutating lookup; expired entries are dropped, live ones untouched.
    if (!id || !byId.has(id)) return null;
    const entry = byId.get(id);
    if (expired(entry, now)) {
      byId.delete(id);
      return null;
    }
    return entry.msg;
  }

  function latestForChat(chatId) {
    if (!chatId) return null;
    // Map preserves insertion order; walk newest-first for the most recent
    // remembered message in this chat (used to react without an explicit id).
    const entries = Array.from(byId.values());
    for (let i = entries.length - 1; i >= 0; i -= 1) {
      if (entries[i]?.msg?.key?.remoteJid === chatId) return entries[i].msg;
    }
    return null;
  }

  function latestInboundForChat(chatId, { now = Date.now(), normalize = normalizeChatJid } = {}) {
    // Most recent RECEIVED message for the chat by seenAt (not LRU position —
    // get()'s touch would otherwise shuffle stale poll lookups to "latest").
    // Outbound (fromMe) and expired entries are skipped.
    const target = normalize(chatId || '');
    if (!target) return null;
    let latest = null;
    for (const entry of byId.values()) {
      if (expired(entry, now)) continue;
      if (entry.msg?.key?.fromMe) continue;
      if (normalize(entry.msg?.key?.remoteJid || '') !== target) continue;
      if (!latest || entry.seenAt >= latest.seenAt) latest = entry;
    }
    return latest ? latest.msg : null;
  }

  return {
    remember,
    get,
    peek,
    latestForChat,
    latestInboundForChat,
    clear: () => byId.clear(),
    get size() {
      return byId.size;
    },
  };
}

/**
 * Bounded cache of the already-downloaded media (and plain text) for
 * recently seen inbound messages, keyed by "chatId:messageId".
 *
 * When a user replies to an earlier message, Baileys' contextInfo.quotedMessage
 * only carries a thumbnail-sized stub for media (or nothing at all for an
 * uncaptioned attachment) — never a way to re-fetch the full original file.
 * Without this cache, replying to a photo/video/document/voice note with no
 * caption gives the agent no text and no media reference: it looks like the
 * message never had an attachment. Since extractBridgeEvent already downloads
 * and caches the media for every inbound message as it arrives, remembering
 * that outcome here lets a later reply resolve the original file path.
 */
export function createQuotedMediaCache(limit = 512) {
  const byKey = new Map();

  function key(chatId, messageId) {
    return `${chatId || ''}:${messageId || ''}`;
  }

  function remember(chatId, messageId, payload) {
    if (!messageId) return;
    const k = key(chatId, messageId);
    byKey.delete(k);
    byKey.set(k, payload);
    while (byKey.size > limit) {
      const oldest = byKey.keys().next().value;
      byKey.delete(oldest);
    }
  }

  function get(chatId, messageId) {
    if (!messageId) return null;
    return byKey.get(key(chatId, messageId)) || null;
  }

  return { remember, get };
}

export function pollCreationMessageSecret(pollCreation) {
  return pollCreation?.message?.messageContextInfo?.messageSecret
    || pollCreation?.messageContextInfo?.messageSecret
    || null;
}

function uniqueStrings(values) {
  const seen = new Set();
  const out = [];
  for (const value of values || []) {
    const text = String(value || '').trim();
    if (!text || seen.has(text)) continue;
    seen.add(text);
    out.push(text);
  }
  return out;
}

export function pollUpdateForAggregation({
  pollUpdateMessage,
  pollUpdateMessageKey,
  pollCreation,
  decryptPollVote,
  getKeyAuthor,
  meId = 'me',
  pollCreatorJids = [],
  voterJids = [],
}) {
  if (!pollUpdateMessage) return null;
  const updateKey = pollUpdateMessage.pollUpdateMessageKey
    || pollUpdateMessageKey
    || pollUpdateMessage.key;
  if (!updateKey) return null;

  if (pollUpdateMessage.vote?.selectedOptions) {
    return {
      pollUpdateMessageKey: updateKey,
      vote: pollUpdateMessage.vote,
      senderTimestampMs: pollUpdateMessage.senderTimestampMs,
    };
  }

  const creationKey = pollUpdateMessage.pollCreationMessageKey;
  const secret = pollCreationMessageSecret(pollCreation);
  if (
    !creationKey?.id
    || !secret
    || !pollUpdateMessage.vote?.encPayload
    || !pollUpdateMessage.vote?.encIv
    || typeof decryptPollVote !== 'function'
    || typeof getKeyAuthor !== 'function'
  ) {
    return null;
  }

  // Baileys poll decryption keys include both creator and voter JIDs.  On
  // WhatsApp LID chats, the poll creator can be the linked-device LID even
  // when sock.user.id is the classic @s.whatsapp.net JID.  Try the exact
  // candidates the live bridge knows before falling back to the generic helper.
  const creatorCandidates = uniqueStrings([
    ...pollCreatorJids,
    getKeyAuthor(creationKey, meId),
  ]);
  const voterCandidates = uniqueStrings([
    ...voterJids,
    getKeyAuthor(updateKey, meId),
  ]);

  let lastError = null;
  for (const pollCreatorJid of creatorCandidates) {
    for (const voterJid of voterCandidates) {
      try {
        const vote = decryptPollVote(pollUpdateMessage.vote, {
          pollCreatorJid,
          pollMsgId: creationKey.id,
          pollEncKey: secret,
          voterJid,
        });
        return {
          pollUpdateMessageKey: updateKey,
          vote,
          senderTimestampMs: pollUpdateMessage.senderTimestampMs,
        };
      } catch (err) {
        lastError = err;
      }
    }
  }
  if (lastError) throw lastError;
  return null;
}

export function addMentions(payload, mentions) {
  if (payload && Array.isArray(mentions) && mentions.length > 0) {
    payload.mentions = mentions;
  }
  return payload;
}

export function buildTextSendPayload(text, { replyTo, messageStore, mentions } = {}) {
  const content = addMentions({ text }, mentions);
  const options = {};
  const quoted = messageStore?.get(replyTo);
  if (quoted?.key && quoted?.message) {
    // Baileys expects quoted messages as sendMessage options, not inside the
    // message content payload. Keeping this split avoids silently sending a
    // literal/ignored `quoted` field instead of a native WhatsApp reply.
    options.quoted = quoted;
  }
  return { content, options };
}

export function buildLocationPayload({ latitude, longitude, name, address } = {}) {
  const lat = Number(latitude);
  const lon = Number(longitude);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
    throw new Error('latitude and longitude must be numbers');
  }
  if (lat < -90 || lat > 90 || lon < -180 || lon > 180) {
    throw new Error('latitude/longitude out of range');
  }

  const location = {
    degreesLatitude: lat,
    degreesLongitude: lon,
  };
  if (name) location.name = String(name);
  if (address) location.address = String(address);
  return { location };
}

function textFromQuotedMessage(quotedMessage) {
  const content = unwrapMessageEnvelopes(quotedMessage);
  if (!content) return '';
  if (content.conversation) return content.conversation;
  if (content.extendedTextMessage?.text) return content.extendedTextMessage.text;
  if (content.imageMessage?.caption) return content.imageMessage.caption;
  if (content.videoMessage?.caption) return content.videoMessage.caption;
  if (content.documentMessage?.caption) return content.documentMessage.caption;
  if (content.documentMessage?.fileName) return `[Document: ${content.documentMessage.fileName}]`;
  if (content.locationMessage) return formatLocationText(content.locationMessage, false);
  if (content.contactMessage) return formatContactText(content.contactMessage);
  if (content.pollCreationMessage) return formatPollText(content.pollCreationMessage);
  return '';
}

function mediaExtForMime(mime, fallback) {
  const normalized = String(mime || '').split(';', 1)[0].toLowerCase();
  const extMap = {
    'image/jpeg': '.jpg',
    'image/png': '.png',
    'image/webp': '.webp',
    'image/gif': '.gif',
    'video/mp4': '.mp4',
    'video/quicktime': '.mov',
    'video/x-matroska': '.mkv',
    'audio/ogg': '.ogg',
    'audio/mp4': '.m4a',
    'audio/mpeg': '.mp3',
    'application/pdf': '.pdf',
  };
  return extMap[normalized] || fallback;
}

function defaultWriteMediaFile({ buffer, dir, prefix, ext, fileName }) {
  mkdirSync(dir, { recursive: true });
  let safeName = fileName ? `_${path.basename(fileName).replace(/[^a-zA-Z0-9._-]/g, '_')}` : '';
  if (safeName && ext && !path.extname(safeName)) {
    safeName = `${safeName}${ext}`;
  }
  const filePath = path.join(dir, `${prefix}_${randomBytes(6).toString('hex')}${safeName || ext}`);
  writeFileSync(filePath, buffer);
  return filePath;
}

function formatLocationText(location, isLive) {
  const name = location.name || location.address || '';
  const lat = location.degreesLatitude ?? location.latitude;
  const lng = location.degreesLongitude ?? location.longitude;
  const kind = isLive ? 'Live location' : 'Location';
  const coords = lat !== undefined && lng !== undefined ? `${lat},${lng}` : '';
  return `[${kind}: ${[name, coords].filter(Boolean).join(' ')}]`;
}

function locationMetadata(location, isLive) {
  return {
    name: location.name || '',
    address: location.address || '',
    latitude: location.degreesLatitude ?? location.latitude ?? null,
    longitude: location.degreesLongitude ?? location.longitude ?? null,
    isLive,
  };
}

function formatContactText(contact) {
  const name = contact.displayName || contact.vcard?.match(/FN:(.+)/)?.[1] || 'unknown';
  const phone = contact.vcard?.match(/TEL[^:]*:(.+)/)?.[1] || '';
  return `[Contact: ${[name, phone].filter(Boolean).join(' ')}]`;
}

function formatContactsText(contacts) {
  const names = contacts.map(c => c.displayName).filter(Boolean);
  return `[Contacts: ${names.join(', ') || contacts.length}]`;
}

function formatReactionText(reaction) {
  const emoji = reaction.text || '';
  const target = reaction.key?.id || '';
  return `[Reaction: ${emoji}${target ? ` to ${target}` : ''}]`;
}

function pollOptions(poll) {
  return (poll.options || [])
    .map(option => option.optionName || option.name)
    .filter(Boolean);
}

function formatPollText(poll) {
  const question = poll.name || poll.title || 'poll';
  const options = pollOptions(poll);
  return `[Poll: ${question}${options.length ? ` Options: ${options.join(', ')}` : ''}]`;
}

function formatPollUpdateText(update) {
  const target = update.pollCreationMessageKey?.id || update.key?.id || '';
  return `[Poll update${target ? `: ${target}` : ''}]`;
}

/**
 * Append a visible note for media that failed to download, so the agent knows
 * something was sent rather than silently losing the attachment. Returns
 * `content` unchanged when nothing failed. (Port of nanoclaw#2895.)
 */
export function appendMediaFailureNote(content, failures) {
  if (!failures || failures.length === 0) return content;
  const note = failures.map((t) => `[${t} could not be downloaded]`).join(' ');
  return content ? `${content}\n${note}` : note;
}

export async function extractBridgeEvent({
  msg,
  chatId,
  senderId,
  senderNumber,
  botIds = [],
  isGroup = false,
  downloadMedia,
  writeMediaFile,
  cacheDirs = {},
  lookupQuotedMedia,
}) {
  const messageContent = getMessageContent(msg);
  const contextInfo = getContextInfo(messageContent);
  const mentionedIds = Array.from(new Set((contextInfo?.mentionedJid || []).map(normalizeWhatsAppId).filter(Boolean)));
  const quotedMessageId = contextInfo?.stanzaId || null;
  const quotedParticipant = normalizeWhatsAppId(contextInfo?.participant || '') || null;
  const quotedRemoteJid = normalizeWhatsAppId(contextInfo?.remoteJid || '') || null;
  const hasQuotedMessage = !!contextInfo?.quotedMessage;
  let quotedText = textFromQuotedMessage(contextInfo?.quotedMessage);
  let quotedMediaUrls = [];
  let quotedMediaType = '';

  // contextInfo.quotedMessage only ever carries a thumbnail-sized stub for
  // media (or nothing for an uncaptioned attachment) — never a way to
  // re-fetch the original file. Resolve the quoted message's already-cached
  // media (downloaded when it first arrived) via lookupQuotedMedia so a
  // reply to an uncaptioned photo/video/document/voice note still gives the
  // agent the original file, not just silence.
  if (quotedMessageId && typeof lookupQuotedMedia === 'function') {
    const original = lookupQuotedMedia(quotedRemoteJid || chatId, quotedMessageId);
    if (original) {
      if (original.hasMedia && original.mediaUrls?.length) {
        quotedMediaUrls = original.mediaUrls;
        quotedMediaType = original.mediaType || '';
        if (!quotedText) {
          quotedText = {
            image: 'sent an image',
            video: 'sent a video',
            gif: 'sent a GIF',
            audio: 'sent an audio message',
            ptt: 'sent a voice message',
            document: 'sent a document',
            sticker: 'sent a sticker',
          }[original.mediaType] || 'sent media';
        }
      } else if (original.body && !quotedText) {
        quotedText = original.body;
      }
    }
  }

  let body = '';
  let hasMedia = false;
  let mediaType = '';
  let mime = '';
  let fileName = '';
  let nativeType = '';
  const mediaUrls = [];
  const nativeMetadata = {};

  const mediaFailures = [];

  const saveMedia = async ({ mediaMessage, dir, prefix, fallbackExt, fileName: name, type }) => {
    if (!downloadMedia) return;
    try {
      const buf = await downloadMedia(msg);
      const ext = mediaExtForMime(mediaMessage?.mimetype, fallbackExt);
      const writer = writeMediaFile || defaultWriteMediaFile;
      const saved = await writer({ buffer: buf, dir, prefix, ext, fileName: name });
      if (saved) mediaUrls.push(saved);
    } catch (err) {
      // A failed CDN fetch (expired media URL, transient network error) must
      // never reject out of extractBridgeEvent — that would drop this message
      // AND every remaining message in the same upsert batch. Record the
      // failure so the agent is told media was sent instead of losing it
      // silently. (Port of nanoclaw#2895's never-silently-drop guarantee; the
      // reuploadRequest recovery half is already wired in bridge.js.)
      mediaFailures.push(type || 'media');
      try {
        console.warn(`[bridge] failed to download inbound ${type || 'media'}:`, err?.message || err);
      } catch {}
    }
  };

  if (messageContent.conversation) {
    body = messageContent.conversation;
    nativeType = 'conversation';
  } else if (messageContent.extendedTextMessage?.text) {
    body = messageContent.extendedTextMessage.text;
    nativeType = 'extendedTextMessage';
  } else if (messageContent.imageMessage) {
    const item = messageContent.imageMessage;
    body = item.caption || '';
    hasMedia = true;
    mediaType = 'image';
    nativeType = 'imageMessage';
    mime = item.mimetype || 'image/jpeg';
    await saveMedia({ mediaMessage: item, dir: cacheDirs.image, prefix: 'img', fallbackExt: '.jpg', type: 'image' });
  } else if (messageContent.videoMessage) {
    const item = messageContent.videoMessage;
    body = item.caption || '';
    hasMedia = true;
    mediaType = item.gifPlayback ? 'gif' : 'video';
    nativeType = 'videoMessage';
    mime = item.mimetype || 'video/mp4';
    nativeMetadata.video = { gifPlayback: !!item.gifPlayback };
    await saveMedia({ mediaMessage: item, dir: cacheDirs.document, prefix: 'vid', fallbackExt: '.mp4', type: mediaType });
  } else if (messageContent.audioMessage || messageContent.pttMessage) {
    const item = messageContent.pttMessage || messageContent.audioMessage;
    hasMedia = true;
    mediaType = item.ptt || messageContent.pttMessage ? 'ptt' : 'audio';
    nativeType = messageContent.pttMessage ? 'pttMessage' : 'audioMessage';
    mime = item.mimetype || 'audio/ogg';
    nativeMetadata.audio = { ptt: mediaType === 'ptt' };
    await saveMedia({ mediaMessage: item, dir: cacheDirs.audio, prefix: 'aud', fallbackExt: '.ogg', type: 'audio' });
  } else if (messageContent.documentMessage) {
    const item = messageContent.documentMessage;
    body = item.caption || '';
    hasMedia = true;
    mediaType = 'document';
    nativeType = 'documentMessage';
    mime = item.mimetype || 'application/octet-stream';
    fileName = item.fileName || 'document';
    await saveMedia({ mediaMessage: item, dir: cacheDirs.document, prefix: 'doc', fallbackExt: '.bin', fileName, type: 'document' });
  } else if (messageContent.stickerMessage) {
    hasMedia = true;
    mediaType = 'sticker';
    nativeType = 'stickerMessage';
    mime = messageContent.stickerMessage.mimetype || 'image/webp';
    body = '[Sticker]';
    nativeMetadata.sticker = {
      animated: !!messageContent.stickerMessage.isAnimated,
      mimetype: mime,
    };
    await saveMedia({ mediaMessage: messageContent.stickerMessage, dir: cacheDirs.image, prefix: 'sticker', fallbackExt: '.webp', type: 'sticker' });
  } else if (messageContent.locationMessage || messageContent.liveLocationMessage) {
    const isLive = !!messageContent.liveLocationMessage;
    const item = messageContent.liveLocationMessage || messageContent.locationMessage;
    mediaType = isLive ? 'live_location' : 'location';
    nativeType = isLive ? 'liveLocationMessage' : 'locationMessage';
    body = formatLocationText(item, isLive);
    nativeMetadata.location = locationMetadata(item, isLive);
  } else if (messageContent.contactMessage) {
    mediaType = 'contact';
    nativeType = 'contactMessage';
    body = formatContactText(messageContent.contactMessage);
    nativeMetadata.contact = {
      displayName: messageContent.contactMessage.displayName || '',
      vcard: messageContent.contactMessage.vcard || '',
    };
  } else if (messageContent.contactsArrayMessage) {
    const contacts = messageContent.contactsArrayMessage.contacts || [];
    mediaType = 'contacts';
    nativeType = 'contactsArrayMessage';
    body = formatContactsText(contacts);
    nativeMetadata.contacts = contacts.map(contact => ({
      displayName: contact.displayName || '',
      vcard: contact.vcard || '',
    }));
  } else if (messageContent.reactionMessage) {
    mediaType = 'reaction';
    nativeType = 'reactionMessage';
    body = formatReactionText(messageContent.reactionMessage);
    nativeMetadata.reaction = {
      text: messageContent.reactionMessage.text || '',
      messageId: messageContent.reactionMessage.key?.id || '',
      remoteJid: normalizeWhatsAppId(messageContent.reactionMessage.key?.remoteJid || ''),
      participant: normalizeWhatsAppId(messageContent.reactionMessage.key?.participant || ''),
    };
  } else if (messageContent.pollCreationMessage || messageContent.pollCreationMessageV2 || messageContent.pollCreationMessageV3) {
    const item = messageContent.pollCreationMessage || messageContent.pollCreationMessageV2 || messageContent.pollCreationMessageV3;
    mediaType = 'poll';
    nativeType = messageContent.pollCreationMessage ? 'pollCreationMessage' : messageContent.pollCreationMessageV2 ? 'pollCreationMessageV2' : 'pollCreationMessageV3';
    body = formatPollText(item);
    nativeMetadata.poll = {
      question: item.name || item.title || '',
      options: pollOptions(item),
      selectableCount: item.selectableOptionsCount || item.selectableCount || 1,
    };
  } else if (messageContent.pollUpdateMessage) {
    mediaType = 'poll_update';
    nativeType = 'pollUpdateMessage';
    body = formatPollUpdateText(messageContent.pollUpdateMessage);
    nativeMetadata.pollUpdate = messageContent.pollUpdateMessage;
  }

  // Surface failed downloads to the agent instead of silently losing the
  // attachment. Applied before the generic "[<type> received]" fallback so an
  // uncaptioned message whose download failed reads "[image could not be
  // downloaded]" rather than claiming the media arrived.
  body = appendMediaFailureNote(body, mediaFailures);

  if (hasMedia && !body) {
    body = `[${mediaType} received]`;
  }

  return {
    messageId: msg.key.id,
    chatId,
    senderId,
    senderName: msg.pushName || senderNumber,
    chatName: isGroup ? (chatId.split('@')[0]) : (msg.pushName || senderNumber),
    isGroup,
    body,
    hasMedia,
    mediaType,
    mime,
    fileName,
    nativeType,
    nativeMetadata,
    mediaUrls,
    mentionedIds,
    quotedMessageId,
    quotedParticipant,
    quotedRemoteJid,
    quotedText,
    quotedMediaUrls,
    quotedMediaType,
    hasQuotedMessage,
    botIds,
    readReceiptKey: {
      remoteJid: msg.key.remoteJid || chatId,
      id: msg.key.id,
      // Groups only: a participant on a DM receipt key makes WhatsApp drop it.
      ...(isGroup ? { participant: msg.key.participant || senderId } : {}),
      fromMe: Boolean(msg.key.fromMe),
    },
    timestamp: msg.messageTimestamp,
  };
}

/** HTTP statuses that mean "this media URL is dead, ask WhatsApp to re-upload".
 *
 * Baileys' own retry set is [410, 404], but WhatsApp's CDN answers an expired
 * `oe=` deadline with **403** — verified live against a real expired sticker
 * URL from bridge.log. Without 403 the recovery path is unreachable for the
 * most common expiry case.
 */
const REUPLOAD_RECOVERABLE_STATUS = new Set([403, 404, 410]);

/** Extract an HTTP status from an error, wherever the thrower put it.
 *
 * @hapi/boom (what Baileys throws) exposes the code at `output.statusCode`;
 * `error.status` is undefined. Baileys' internal recovery only reads
 * `error.status`, which is why its reupload path never fires in practice.
 */
export function httpStatusFromError(err) {
  const candidates = [
    err?.status,
    err?.statusCode,
    err?.output?.statusCode,
    err?.output?.payload?.statusCode,
  ];
  for (const value of candidates) {
    if (typeof value === 'number' && Number.isFinite(value)) return value;
  }
  return null;
}

/**
 * Download inbound media, recovering from an expired CDN URL.
 *
 * WhatsApp media URLs carry an `oe=` expiry; once past it the CDN returns 403
 * and the bytes are only retrievable by asking WhatsApp to re-upload the
 * message (`sock.updateMediaMessage`). Baileys ships that recovery but gates
 * it on `error.status` ∈ {410, 404} — a field @hapi/boom never sets, for a
 * status the CDN never returns. Both halves miss, so every expired attachment
 * degraded to "[sticker could not be downloaded]".
 *
 * We retry exactly once: request the re-upload, then re-attempt the download.
 * A second failure, or any non-HTTP error (bad decrypt, disk full), propagates
 * to the caller unchanged — re-uploading cannot fix those and looping would
 * hammer the socket.
 */
export async function downloadMediaWithRecovery({ msg, download, reuploadRequest }) {
  try {
    return await download(msg);
  } catch (err) {
    const status = httpStatusFromError(err);
    if (
      typeof reuploadRequest !== 'function'
      || status === null
      || !REUPLOAD_RECOVERABLE_STATUS.has(status)
    ) {
      throw err;
    }
    const refreshed = await reuploadRequest(msg);
    return await download(refreshed || msg);
  }
}

export function inferMediaType(ext) {
  if (['jpg', 'jpeg', 'png', 'webp', 'gif'].includes(ext)) return 'image';
  if (['mp4', 'mov', 'avi', 'mkv', '3gp'].includes(ext)) return 'video';
  if (['ogg', 'opus', 'mp3', 'wav', 'm4a'].includes(ext)) return 'audio';
  return 'document';
}

export function inboundReadReceiptKeys({ key, enabled }) {
  if (!enabled || !key || key.fromMe || !key.id || !key.remoteJid) return [];
  // Preserve participant for group messages: Baileys needs the original key.
  if (key.remoteJid.endsWith('@g.us')) return [key];
  // A DM key must not carry participant. Baileys' sendReceipt copies it onto the
  // <receipt> node, which then looks group-shaped and WhatsApp drops it: no blue
  // ticks, yet readMessages() resolves and /read reports marked:true.
  const { participant: _participant, ...dmKey } = key;
  return [dmKey];
}

export function mediaPayloadForFile({ buffer, filePath, mediaType, caption, fileName }) {
  const ext = filePath.toLowerCase().split('.').pop();
  const type = mediaType || inferMediaType(ext);
  if (type === 'image' && ext === 'gif') {
    // Pure helper fallback: do not lie and label raw GIF bytes as mp4.
    // The live bridge tries ffmpeg conversion to WhatsApp gifPlayback video
    // before it falls back to this regular image payload.
    return { image: buffer, caption: caption || undefined, mimetype: MIME_MAP[ext] || 'image/gif' };
  }
  switch (type) {
    case 'image':
      return { image: buffer, caption: caption || undefined, mimetype: MIME_MAP[ext] || 'image/jpeg' };
    case 'video':
      return { video: buffer, caption: caption || undefined, mimetype: MIME_MAP[ext] || 'video/mp4' };
    case 'sticker':
      // Explicit [[as_sticker]] directive: raw bytes as a WhatsApp sticker.
      // Stickers carry no caption bubble.
      return { sticker: buffer };
    case 'document':
      return {
        document: buffer,
        fileName: fileName || path.basename(filePath),
        caption: caption || undefined,
        mimetype: MIME_MAP[ext] || 'application/octet-stream',
      };
    default:
      return null;
  }
}

export function buildPollPayload({ question, options, selectableCount = 1 }) {
  const cleanQuestion = String(question || '').trim();
  const cleanOptions = (options || []).map(option => String(option || '').trim()).filter(Boolean);
  if (!cleanQuestion) throw new Error('question is required');
  if (cleanOptions.length < 2) throw new Error('at least two poll options are required');
  if (cleanOptions.length > 12) throw new Error('at most 12 poll options are supported');
  const count = Math.max(1, Math.min(Number(selectableCount) || 1, cleanOptions.length));
  return {
    poll: {
      name: cleanQuestion,
      values: cleanOptions,
      selectableCount: count,
      messageSecret: randomBytes(32),
    },
  };
}

export function pollCreationMessageFromPayload(payload) {
  const poll = payload?.poll;
  if (!poll) return null;
  const values = Array.isArray(poll.values) ? poll.values : [];
  const options = values.map(value => String(value || '').trim()).filter(Boolean);
  if (!poll.name || options.length < 2) return null;
  const selectableOptionsCount = Math.max(1, Math.min(Number(poll.selectableCount) || 1, options.length));
  const message = {};
  if (poll.messageSecret) {
    message.messageContextInfo = { messageSecret: poll.messageSecret };
  }
  message[selectableOptionsCount === 1 ? 'pollCreationMessageV3' : 'pollCreationMessage'] = {
    name: String(poll.name),
    options: options.map(optionName => ({ optionName })),
    selectableOptionsCount,
  };
  return message;
}

/**
 * Reconnect scheduling guard. startSocket() awaits network I/O before it
 * creates a socket or registers event handlers, so a bare
 * `setTimeout(startSocket, ...)` has two unrecoverable failure modes: a
 * rejection is unhandled (crashes the process on modern Node), and a hang
 * leaves the bridge permanently disconnected with nothing left to retry.
 * Every (re)connect must go through the scheduler this returns.
 */
export function createReconnectScheduler(startFn, {
  retryDelayMs = 5000,
  log = console.log,
  setTimeoutFn = setTimeout,
} = {}) {
  function scheduleReconnect(delayMs) {
    setTimeoutFn(() => {
      Promise.resolve()
        .then(startFn)
        .catch((err) => {
          log(`⚠️  Reconnect failed (${err?.message || err}). Retrying in ${Math.round(retryDelayMs / 1000)}s...`);
          scheduleReconnect(retryDelayMs);
        });
    }, delayMs);
  }
  return scheduleReconnect;
}

/**
 * Version resolution guard. fetchLatestBaileysVersion() is a plain fetch to
 * raw.githubusercontent.com with no AbortSignal; a stalled connection can
 * pend forever and wedge the reconnect path (the scheduler above cannot
 * retry past an await that never settles). Bound the fetch and fall back to
 * the last known-good version, or the Baileys default before first success.
 */
export function createVersionResolver(fetchVersionFn, {
  timeoutMs = 15000,
  log = console.log,
} = {}) {
  let cachedVersion = null;
  return async function resolveVersion() {
    let timer = null;
    try {
      const { version } = await Promise.race([
        fetchVersionFn(),
        new Promise((_, reject) => {
          timer = setTimeout(() => reject(new Error('version fetch timed out')), timeoutMs);
        }),
      ]);
      cachedVersion = version;
    } catch (err) {
      log(`⚠️  Baileys version fetch failed (${err?.message || err}); using ${cachedVersion ? 'cached version' : 'library default'}.`);
    } finally {
      if (timer) clearTimeout(timer);
    }
    return cachedVersion;
  };
}

// ── Restored WhatsApp native surface helpers (ported from live ref 828ce02edf) ──

/**
 * Normalize a JID for *chat identity* comparison.
 *
 * Distinct from ``normalizeWhatsAppId`` (which rewrites ``:`` to ``@`` for id
 * display): this strips the device/agent suffix so ``1555:12@s.whatsapp.net``
 * and ``1555@s.whatsapp.net`` are recognized as the same conversation.
 *
 * The server domain is deliberately PRESERVED. ``@lid`` is a separate address
 * space from ``@s.whatsapp.net`` — the numeric part of a LID is not a phone
 * number — so the two must never compare equal, or a quote from one chat could
 * be attached to a send in another.
 */
export function normalizeChatJid(value) {
  if (!value) return '';
  const raw = String(value).trim().toLowerCase();
  if (!raw) return '';
  const at = raw.lastIndexOf('@');
  if (at === -1) return raw.split(':')[0];
  const user = raw.slice(0, at).split(':')[0];
  const server = raw.slice(at + 1);
  if (!user || !server) return '';
  return `${user}@${server}`;
}

/**
 * Resolve who a cached status/story belongs to. A status message's own key
 * carries the author as `participant` (group-style addressing even though
 * status@broadcast isn't a group); `statusAuthorJid` from the caller is an
 * explicit override for when the cache lacks participant metadata (e.g. the
 * status was seen before the bridge process restarted and repopulated from a
 * fresh upsert that happened to omit it).
 */
export function getStatusAuthor(statusMsg, fallback) {
  return normalizeWhatsAppId(
    fallback || statusMsg?.key?.participant || statusMsg?.participant || ''
  );
}

/**
 * Build the WAMessageKey `sendMessage(jid, { delete: key })` needs to delete
 * a message for everyone. `fromMe` must be explicit and boolean — deleting
 * someone else's message in a group requires admin rights server-side and a
 * wrong `fromMe` silently no-ops rather than erroring, so this refuses to
 * guess.
 */

/**
 * Build the WAMessageKey `sendMessage(jid, { delete: key })` needs to delete
 * a message for everyone. `fromMe` must be explicit and boolean — deleting
 * someone else's message in a group requires admin rights server-side and a
 * wrong `fromMe` silently no-ops rather than erroring, so this refuses to
 * guess.
 */
export function buildDeleteKey({ chatId, messageId, fromMe, participant }) {
  if (!chatId || !messageId || typeof fromMe !== 'boolean') return null;
  const key = { remoteJid: chatId, id: messageId, fromMe };
  if (participant) key.participant = participant;
  return key;
}

/**
 * Build the `chatModify({ star: {...} }, jid)` payload. `messages` is a list
 * of `{id, fromMe}` — WhatsApp's star primitive is keyed on message
 * ownership, not just id, same reasoning as buildDeleteKey.
 */

/**
 * Build the `chatModify({ star: {...} }, jid)` payload. `messages` is a list
 * of `{id, fromMe}` — WhatsApp's star primitive is keyed on message
 * ownership, not just id, same reasoning as buildDeleteKey.
 */
export function buildStarPayload({ messages, star }) {
  if (!Array.isArray(messages) || messages.length === 0) return null;
  const clean = messages
    .filter((m) => m && m.id && typeof m.fromMe === 'boolean')
    .map((m) => ({ id: String(m.id), fromMe: m.fromMe }));
  if (clean.length === 0) return null;
  return { star: { messages: clean, star: Boolean(star) } };
}

/** `chatModify({ archive, lastMessages }, jid)` — archive/unarchive a chat. */

/** `chatModify({ archive, lastMessages }, jid)` — archive/unarchive a chat. */
export function buildArchivePayload({ archive, lastMessage }) {
  if (typeof archive !== 'boolean' || !lastMessage?.key) return null;
  return { archive, lastMessages: [lastMessage] };
}

/**
 * `chatModify({ mute }, jid)` — mute for a duration in ms, or `null` to
 * unmute. WhatsApp only accepts a small set of durations (8h/7d) server-side
 * but passes through whatever we send; validation of the exact allowed set
 * is left to WhatsApp's own response rather than duplicated here.
 */

/**
 * `chatModify({ mute }, jid)` — mute for a duration in ms, or `null` to
 * unmute. WhatsApp only accepts a small set of durations (8h/7d) server-side
 * but passes through whatever we send; validation of the exact allowed set
 * is left to WhatsApp's own response rather than duplicated here.
 */
export function buildMutePayload({ muteMs }) {
  if (muteMs !== null && (!Number.isFinite(muteMs) || muteMs <= 0)) return null;
  return { mute: muteMs };
}

/** `chatModify({ pin }, jid)` — pin/unpin a chat. */

/** `chatModify({ pin }, jid)` — pin/unpin a chat. */
export function buildPinPayload({ pin }) {
  if (typeof pin !== 'boolean') return null;
  return { pin };
}

/** `chatModify({ markRead, lastMessages }, jid)` — mark a chat read/unread. */

/** `chatModify({ markRead, lastMessages }, jid)` — mark a chat read/unread. */
export function buildMarkReadPayload({ read, lastMessage }) {
  if (typeof read !== 'boolean' || !lastMessage?.key) return null;
  return { markRead: read, lastMessages: [lastMessage] };
}

const _GROUP_SETTINGS = new Set(['announcement', 'not_announcement', 'locked', 'unlocked']);

/** Validate a `groupSettingUpdate` tag against the set Baileys/WA accept. */

/** Validate a `groupSettingUpdate` tag against the set Baileys/WA accept. */
export function isValidGroupSetting(setting) {
  return _GROUP_SETTINGS.has(String(setting || ''));
}

const _GROUP_PARTICIPANT_ACTIONS = new Set(['add', 'remove', 'promote', 'demote']);

/** Validate a `groupParticipantsUpdate` action. */

/** Validate a `groupParticipantsUpdate` action. */
export function isValidGroupParticipantAction(action) {
  return _GROUP_PARTICIPANT_ACTIONS.has(String(action || ''));
}

/**
 * Normalize a phone number for `onWhatsApp` existence checks. Accepts a bare
 * E.164-ish number ("+52155...", "52155...") or an already-built JID and
 * returns the bare-number JID `onWhatsApp` expects — it does its own
 * `+`-stripping internally but rejects LIDs outright, so normalizing here
 * keeps the HTTP layer from ever forwarding a `@lid` into a call that will
 * silently skip it (Baileys logs a warning and returns nothing for that
 * entry, which reads as "not on WhatsApp" if the caller isn't watching for
 * it).
 */

/**
 * Normalize a phone number for `onWhatsApp` existence checks. Accepts a bare
 * E.164-ish number ("+52155...", "52155...") or an already-built JID and
 * returns the bare-number JID `onWhatsApp` expects — it does its own
 * `+`-stripping internally but rejects LIDs outright, so normalizing here
 * keeps the HTTP layer from ever forwarding a `@lid` into a call that will
 * silently skip it (Baileys logs a warning and returns nothing for that
 * entry, which reads as "not on WhatsApp" if the caller isn't watching for
 * it).
 */
export function normalizePhoneForCheck(value) {
  const raw = String(value || '').trim();
  if (!raw) return null;
  if (raw.endsWith('@lid')) return null;
  const digits = raw.includes('@') ? raw.split('@')[0].split(':')[0] : raw.replace(/[^\d]/g, '');
  if (!digits) return null;
  return `${digits}@s.whatsapp.net`;
}

// --- Capability-audit tier 2 (2026-08-11): privacy, blocklist, contacts,
// group-join-approval, ephemeral messages, presence. Same pure-builder
// pattern as tier 1 — every payload validated/shaped here, wired in
// bridge.js only as thin sock.* passthroughs.

const _PRIVACY_CATEGORIES = {
  // category -> {sockMethod, allowedValues}. messages/calladd/groupadd have
  // a narrower value set than the rest (no 'none' — WhatsApp doesn't offer
  // "nobody can message me" as a privacy category, only block does that).
  last: { values: ['all', 'contacts', 'contact_blacklist', 'none'] },
  online: { values: ['all', 'match_last_seen'] },
  profile: { values: ['all', 'contacts', 'contact_blacklist', 'none'] },
  status: { values: ['all', 'contacts', 'contact_blacklist', 'none'] },
  readreceipts: { values: ['all', 'none'] },
  groupadd: { values: ['all', 'contacts', 'contact_blacklist'] },
  calladd: { values: ['all', 'known'] },
  messages: { values: ['all', 'contacts'] },
};

/** Validate a privacy category + value pair against WhatsApp's real enums. */

/** Validate a privacy category + value pair against WhatsApp's real enums. */
export function isValidPrivacyUpdate(category, value) {
  const spec = _PRIVACY_CATEGORIES[String(category || '')];
  if (!spec) return false;
  return spec.values.includes(String(value || ''));
}

/** Map a privacy category name to the exact `sock.update*Privacy` method. */

/** Map a privacy category name to the exact `sock.update*Privacy` method. */
export function privacySockMethodName(category) {
  const map = {
    last: 'updateLastSeenPrivacy',
    online: 'updateOnlinePrivacy',
    profile: 'updateProfilePicturePrivacy',
    status: 'updateStatusPrivacy',
    readreceipts: 'updateReadReceiptsPrivacy',
    groupadd: 'updateGroupsAddPrivacy',
    calladd: 'updateCallPrivacy',
    messages: 'updateMessagesPrivacy',
  };
  return map[String(category || '')] || null;
}

const _BLOCK_ACTIONS = new Set(['block', 'unblock']);

/** Validate a block/unblock action. */

/** Validate a block/unblock action. */
export function isValidBlockAction(action) {
  return _BLOCK_ACTIONS.has(String(action || ''));
}

/**
 * Build the `chatModify({ contact }, jid)` payload for addOrEditContact, or
 * `{ contact: null }` for removeContact. A contact needs at least a name to
 * be worth saving — an empty add is a caller bug, not a valid "clear" (use
 * remove=true for that).
 */

/**
 * Build the `chatModify({ contact }, jid)` payload for addOrEditContact, or
 * `{ contact: null }` for removeContact. A contact needs at least a name to
 * be worth saving — an empty add is a caller bug, not a valid "clear" (use
 * remove=true for that).
 */
export function buildContactPayload({ name, remove }) {
  if (remove) return { contact: null };
  const trimmed = String(name || '').trim();
  if (!trimmed) return null;
  return { contact: { name: trimmed } };
}

const _EPHEMERAL_SECONDS = new Set([0, 86400, 604800, 7776000]);

/** Validate a disappearing-messages duration against WhatsApp's allowed set. */

/** Validate a disappearing-messages duration against WhatsApp's allowed set. */
export function isValidEphemeralSeconds(seconds) {
  return _EPHEMERAL_SECONDS.has(Number(seconds));
}

const _JOIN_APPROVAL_MODES = new Set(['on', 'off']);

/** Validate a group join-approval mode. */

/** Validate a group join-approval mode. */
export function isValidJoinApprovalMode(mode) {
  return _JOIN_APPROVAL_MODES.has(String(mode || ''));
}

const _MEMBER_ADD_MODES = new Set(['all_member_add', 'admin_add']);

/** Validate a group member-add mode. */

/** Validate a group member-add mode. */
export function isValidMemberAddMode(mode) {
  return _MEMBER_ADD_MODES.has(String(mode || ''));
}

const _JOIN_REQUEST_ACTIONS = new Set(['approve', 'reject']);

/** Validate a group join-request approve/reject action. */

/** Validate a group join-request approve/reject action. */
export function isValidJoinRequestAction(action) {
  return _JOIN_REQUEST_ACTIONS.has(String(action || ''));
}

const _PRESENCE_TYPES = new Set(['available', 'unavailable', 'composing', 'recording', 'paused']);

/** Validate a presence-update type. `chatId` is required for the four
 * per-chat types (composing/recording/paused need one to mean anything;
 * `available`/`unavailable` are account-wide but Baileys still accepts an
 * optional target). */

/** Validate a presence-update type. `chatId` is required for the four
 * per-chat types (composing/recording/paused need one to mean anything;
 * `available`/`unavailable` are account-wide but Baileys still accepts an
 * optional target). */
export function isValidPresenceType(type) {
  return _PRESENCE_TYPES.has(String(type || ''));
}

// --- Capability-audit tier 4 (2026-08-11, fourth pass): link-preview
// privacy chatModify wrapper. presenceSubscribe needs no builder — it's a
// single-argument passthrough (subscribe to a JID's presence updates so
// Baileys starts pushing them over the socket) validated by JID-shape alone.

/** `chatModify({ disableLinkPreviews }, '')` — global (not per-chat) toggle. */

/** `chatModify({ disableLinkPreviews }, '')` — global (not per-chat) toggle. */
export function buildDisableLinkPreviewsPayload({ disabled }) {
  if (typeof disabled !== 'boolean') return null;
  return { disableLinkPreviews: { isPreviewsDisabled: disabled } };
}

/**
 * Force outbound text through a link-preview policy.
 *
 * Baileys generates link previews by fetching the URL in the message text. That
 * is an outbound network request driven by message *content*, and it happens
 * inside sendMessage — after any caller-side scope check has already passed — so
 * a scoped send could still reach a URL the scope should not touch. This marks
 * every plain text send as preview-free (`linkPreview: null`) unless the caller
 * has already supplied a preview object, in which case the content is passed
 * through untouched.
 *
 * Two invariants the tests pin, both easy to regress:
 *   * `linkPreview: null` is explicit — leaving the field undefined lets Baileys
 *     generate the preview, which is the behaviour being prevented.
 *   * the caller's own content object is never mutated; the policy builds a copy.
 */
export function installOutboundTextPolicy(socket) {
  const send = socket.sendMessage.bind(socket);
  socket.sendMessage = (jid, content, options) => send(
    jid,
    typeof content?.text === 'string' && content.linkPreview === undefined
      ? { ...content, linkPreview: null }
      : content,
    options,
  );
}
