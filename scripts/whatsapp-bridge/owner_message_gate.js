/**
 * Pure classifier for the WhatsApp bridge's bot-mode dispatch loop.
 *
 * Centralises the "should this fromMe message be forwarded as fromOwner?"
 * decision so the gate can be unit-tested without spinning up Baileys or
 * the Express server.
 *
 * Lives next to `outbound_ids.js` rather than inline in `bridge.js`
 * because the previous implementation accidentally bypassed the
 * customer-side allowlist when forwarding owner-typed messages — see
 * the regression test in `owner_message_gate.test.mjs`.
 *
 * Caller responsibilities:
 *   - Only invoke in bot mode. Self-chat mode has its own self-chat
 *     pinning logic and must not delegate here.
 *   - Status JIDs still need a caller-side pre-filter; groups are handled here, gated
 *     on the group allowlist via `groupAllowed`.
 *   - On `drop_allowlist`, log the rejection so operators can audit
 *     accidental allowlist mismatches.
 *
 * Returned actions:
 *   - 'pass'           : non-fromMe, fall through to existing handling
 *   - 'drop_echo'      : fromMe and matches a recently-sent /send id
 *   - 'drop_disabled'  : fromMe DM but operator hasn't opted into forwarding
 *   - 'drop_allowlist' : fromMe DM whose *customer chatId* isn't on the DM allowlist
 *                        (owner-typed reply to a stranger), OR any fromMe group whose
 *                        chatId isn't on the group allowlist
 *   - 'forward_owner'  : fromMe, owner-typed, allowlisted — forward with
 *                        fromOwner: true
 */

export function classifyOwnerMessageGate({
  fromMe,
  fromOwnerEnabled,
  recentlySent,
  allowlistMatches,
  messageId,
  chatId,
  isGroup = false,
  groupAllowed = false,
}) {
  if (!fromMe) {
    return { action: 'pass' };
  }
  if (recentlySent && recentlySent.has(messageId)) {
    return { action: 'drop_echo' };
  }
  // In a group the consent boundary is the GROUP allowlist, not the DM opt-in: listing a
  // group already means "serve this conversation", and every other member's messages are
  // forwarded there. Gating the owner's own messages on WHATSAPP_FORWARD_OWNER_MESSAGES
  // instead made them the only traffic in an allowlisted group that never arrived.
  if (isGroup) {
    return groupAllowed ? { action: 'forward_owner' } : { action: 'drop_allowlist' };
  }
  if (!fromOwnerEnabled) {
    return { action: 'drop_disabled' };
  }
  // Allowlist gate: check the *customer* chatId, not the sender. The
  // sender is the owner's own number/LID and won't be on the allowlist
  // by construction. Without this check, any contact the owner happens
  // to reply to leaks into Hermes and triggers implicit handover in the
  // gateway-policy plugin.
  if (typeof allowlistMatches === 'function' && !allowlistMatches(chatId)) {
    return { action: 'drop_allowlist' };
  }
  return { action: 'forward_owner' };
}
