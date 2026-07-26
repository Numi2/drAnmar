# Security policy

## Reporting a vulnerability

Do not disclose an exploitable vulnerability in a public issue. Use the repository host's private
security-advisory feature or contact the maintainers privately through the hosting organization. Include
the affected revision, reproduction steps, impact, and any proposed mitigation. Avoid attaching real
patient data, secrets, or harmful payloads.

## Deployment boundary

Dr.Anmar is research software, not a hardened internet service. The hub and worker:

- bind only to `127.0.0.1` by default;
- support access-token authentication and a single-operator mutation lease;
- require an operator identity for every state-changing request, including
  direct API requests that omit the browser `Origin` header;
- reject state-changing browser requests whose Origin host differs from the service host;
- return no-store, no-sniff, no-referrer, and same-site resource headers;
- launch bounded local simulation/training commands; and
- write demonstrations and state beneath `DR_ANMAR_ROOT`.

Non-loopback binding fails closed unless `DR_ANMAR_ALLOW_REMOTE=1`, a nonempty
`DR_ANMAR_ACCESS_TOKEN`, `DR_ANMAR_TLS_TERMINATED=1`, and
`DR_ANMAR_FIREWALL_CONFIRMED=1` are all set. These switches are explicit
deployment acknowledgements, not automatic TLS or firewall configuration.
Run remote access only on a trusted LAN or private VPN behind a TLS reverse
proxy and host firewall. Do not expose ports 2360 or 2361 directly to the
public internet. The same-host Origin check is CSRF defense in depth; the
access token authenticates the workstation and the short operator lease
serializes mutation, but neither turns this research workstation into a
multi-user authorization service.

## Data and clinical safety

Use synthetic or properly governed research data only. Never place identifiable patient information in
issues, demonstrations, logs, screenshots, or repository history. This software must not connect to or
control physical surgical equipment and must not be used for clinical decisions.

Security support currently follows the `main` branch; older snapshots may not receive fixes.
