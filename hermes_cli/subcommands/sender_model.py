"""Local management of gateway sender model defaults (requires profile DB write access).

No gateway slash command is registered. This CLI command has no separate owner-identity
check: protect the profile's state.db through filesystem and shell access controls.
Credentials are resolved when a turn runs, never persisted here.
"""

import argparse

from gateway.sender_model_preference import owner_sender_key
from hermes_state import SessionDB


def build_sender_model_parser(subparsers):
    parser = subparsers.add_parser("sender-model", help="Manage per-platform sender model defaults (requires profile DB write access)")
    actions = parser.add_subparsers(dest="sender_model_action", required=True)
    setter = actions.add_parser("set")
    setter.add_argument("platform")
    setter.add_argument("sender_id")
    setter.add_argument("provider")
    setter.add_argument("model")
    clear = actions.add_parser("clear")
    clear.add_argument("platform")
    clear.add_argument("sender_id")
    actions.add_parser("list")
    parser.set_defaults(func=cmd_sender_model)


def cmd_sender_model(args):
    from hermes_cli.runtime_provider import resolve_runtime_provider, format_runtime_provider_error
    action = args.sender_model_action
    if action == "set":
        try:
            # Validate before writing, using the very same resolver the gateway uses.
            resolve_runtime_provider(requested=args.provider, target_model=args.model)
        except Exception as exc:
            raise SystemExit(format_runtime_provider_error(exc)) from exc
    db = SessionDB(read_only=action == "list")
    try:
        if action == "list":
            for platform, sender, provider, model in db.list_sender_model_preferences():
                print(f"{platform}\t{sender}\t{provider}\t{model}")
        elif action == "set":
            key = owner_sender_key(args.platform, args.sender_id)
            db.set_sender_model_preference(args.platform, key, args.provider, args.model)
            print("Sender model preference saved")
        elif action == "clear":
            key = owner_sender_key(args.platform, args.sender_id)
            db.set_sender_model_preference(args.platform, key, None, None)
            print("Sender model preference cleared")
    finally:
        db.close()
