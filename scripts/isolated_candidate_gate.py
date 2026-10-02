"""Read-only cutover preflight. Never creates containers or edits owner config.

Inputs are an operator-reviewed manifest and a freshly captured docker inspect
object. Evidence fields are explicit review attestations, NOT proof inferred by
this checker. Capture network/egress/feature receipts independently. Exit 1 on
any missing authority, feature evidence or unsafe container setting. No device
broker is implemented: this candidate supports an empty device grant set only.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

EVIDENCE = ('native_worker', 'namespace_faults', 'root_media', 'async_delivery',
            'clarification', 'native_reaction', 'independent_review', 'egress')


def readiness_errors(manifest, inspection):
    errors = []
    def require(ok, code):
        if not ok:
            errors.append(code)
    if not isinstance(manifest, dict) or not isinstance(inspection, dict):
        return ['invalid-document']
    m, i = manifest, inspection
    require(m.get('schema') == 1, 'manifest-schema')
    require(m.get('cutover_authorized') is True, 'cutover-not-authorized')
    require(isinstance(m.get('domain'), str) and re.fullmatch(r'[a-z0-9][a-z0-9_-]*', m['domain']), 'domain-missing')
    binding = m.get('binding')
    require(isinstance(binding, dict) and set(binding) == {'platform', 'chat_id', 'user_id'}
            and binding.get('platform') in {'telegram', 'whatsapp'}
            and all(isinstance(binding.get(k), str) and binding[k].strip()
                    and '*' not in binding[k] for k in ('chat_id', 'user_id')), 'verified-binding-required')
    require(m.get('device_grants') == [], 'device-broker-not-implemented')
    require(m.get('legacy_context') == 'quarantined-not-imported', 'private-brain-must-not-be-cloned')
    evidence = m.get('evidence')
    for name in EVIDENCE:
        require(isinstance(evidence, dict) and evidence.get(name) == 'verified', 'unverified-' + name)
    image = m.get('image_id', '')
    require(isinstance(image, str) and re.fullmatch(r'sha256:[0-9a-f]{64}', image)
            and i.get('Image') == image, 'image-digest-mismatch')
    cfg, h = i.get('Config', {}), i.get('HostConfig', {})
    if not isinstance(cfg, dict) or not isinstance(h, dict):
        return errors + ['invalid-container-facts']
    require(cfg.get('User') == '10000:10000', 'worker-uid')
    env_rows = cfg.get('Env')
    env = {}
    if isinstance(env_rows, list):
        for row in env_rows:
            if not isinstance(row, str) or '=' not in row:
                errors.append('invalid-worker-environment')
                continue
            key, value = row.split('=', 1)
            if key in env:
                errors.append('duplicate-worker-environment')
            env[key] = value
    require(env.get('HOME') == '/opt/data/home' and env.get('HERMES_HOME') == '/opt/data/profile', 'private-home-layout')
    require(not any(re.search(r'(TOKEN|SECRET|API_KEY|SSH_AUTH_SOCK|DOCKER_HOST)', k) for k in env), 'inherited-authority')
    require(h.get('ReadonlyRootfs') is True and h.get('Privileged') is False, 'readonly-unprivileged-required')
    require(h.get('CapDrop') == ['ALL'] and not h.get('CapAdd'), 'capabilities-required')
    opts = h.get('SecurityOpt', [])
    require(isinstance(opts, list) and any(x in opts for x in ('no-new-privileges', 'no-new-privileges:true'))
            and not any('unconfined' in str(x) for x in opts), 'security-options')
    network = m.get('network')
    require(isinstance(network, str) and network not in {'', 'host', 'bridge', 'none'}
            and h.get('NetworkMode') == network, 'dedicated-internal-network-required')
    require(h.get('PidMode') in ('', None) and h.get('IpcMode') == 'private', 'namespace-sharing')
    require(not h.get('Binds') and not i.get('Mounts'), 'host-or-volume-mount-denied')
    require(not h.get('Devices') and not h.get('DeviceRequests'), 'device-mount-denied')
    require(not h.get('PortBindings'), 'published-listeners-denied')
    require(not h.get('Sysctls'), 'custom-sysctls-denied')
    for field, maximum in (('Memory', 2147483648), ('PidsLimit', 256), ('NanoCpus', 1000000000)):
        value = h.get(field)
        require(type(value) is int and 0 < value <= maximum, 'unbounded-' + field.lower())
    require(h.get('Dns') == ['1.1.1.1'] and h.get('DnsSearch') == ['.'], 'private-dns-inheritance')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--inspect', type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        inspection = json.loads(args.inspect.read_text(encoding="utf-8"))
        if isinstance(inspection, list) and len(inspection) == 1:
            inspection = inspection[0]
        errors = readiness_errors(manifest, inspection)
    except (OSError, ValueError, TypeError):
        errors = ['unreadable-or-invalid-input']
    # Only fixed error codes: never echo config/env/credential values.
    print(json.dumps({'ready': not errors, 'errors': errors}, sort_keys=True))
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
