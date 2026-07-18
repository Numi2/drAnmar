# Security policy

## Reporting a vulnerability

Do not disclose an exploitable vulnerability in a public issue. Use the repository host's private
security-advisory feature or contact the maintainers privately through the hosting organization. Include
the affected revision, reproduction steps, impact, and any proposed mitigation. Avoid attaching real
patient data, secrets, or harmful payloads.

## Deployment boundary

Dr.Anmar is research software, not a hardened internet service. The hub and worker:

- bind to all network interfaces by default;
- do not provide authentication, authorization, or TLS;
- launch bounded local simulation/training commands; and
- write demonstrations and state beneath `DR_ANMAR_ROOT`.

Run them only on a trusted LAN or private VPN. Do not expose ports 2360 or 2361 to the public internet.
For broader access, place the service behind an authenticated TLS reverse proxy and host firewall.

## Data and clinical safety

Use synthetic or properly governed research data only. Never place identifiable patient information in
issues, demonstrations, logs, screenshots, or repository history. This software must not connect to or
control physical surgical equipment and must not be used for clinical decisions.

Security support currently follows the `main` branch; older snapshots may not receive fixes.
