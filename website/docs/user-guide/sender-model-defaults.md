# Sender-scoped gateway model defaults

An operator with local access to the profile's Hermes home can set a persistent model for a **platform sender**, not for a chat or a profile. This does not create another agent: conversations still share the serving profile's personality, tools, credentials, memory, and session database.

```sh
hermes sender-model set discord USER_ID openrouter vendor/model-id
hermes sender-model list
hermes sender-model clear discord USER_ID
```

Run these commands in the same `HERMES_HOME`/profile scope as the gateway's serving profile. They are local-only: **any process with write access to that profile's `state.db` can change the preference**; the command does not authenticate an owner identity. Chat users cannot run `sender-model`, but `/model` is a separate slash command. To restrict `/model` to administrators, configure `allow_admin_from` and omit `model` from `user_allowed_commands` (and their `group_*` counterparts) in each platform's `extra` section; without an admin list, upstream slash-command gating is disabled for that scope. `set` validates the provider/target model through the existing runtime provider resolver before writing. No API key is saved in the preference table; the gateway resolves credentials through the normal provider ladder on each turn. Missing credentials fail at provider resolution rather than silently falling back to another model.

Precedence: saved `/model` session choice (including `--once` turn behavior) > explicit channel model/provider > sender default > profile/global default. Sender lookup uses the author's `user_id`, not `chat_id`; messages without a sender retain previous behavior. Platforms never share a key, even for identical IDs. WhatsApp phone/LID aliases are expanded through the existing bridge mapping at lookup time; conflicting preferences across aliases raise an error instead of choosing a row. A mapping not yet known when a preference is set will work after it becomes available. Preferences are read per turn, with no preference cache; existing cached agents rebuild when the effective model/provider signature changes. Clearing a preference returns the sender to channel/profile precedence on the next turn. For multiplexed profiles, each serving profile's database owns its own preference; target the profile's local home when running the CLI.

The schema is additive (`gateway_sender_model_preferences` in `state.db`); startup `CREATE TABLE IF NOT EXISTS` migrates existing databases. There is no cross-platform identity linking. Do not use this feature as authorization: gateway admission and slash-command policies still decide whether a sender can reach the agent or switch its model.
