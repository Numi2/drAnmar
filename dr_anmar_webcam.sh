#!/usr/bin/env bash

set -euo pipefail

COMMAND="${1:-start}"
SSH_TARGET="${DR_ANMAR_SSH_TARGET:-numi@100.98.17.98}"
LOCAL_PORT="${DR_ANMAR_LOCAL_PORT:-12360}"
REMOTE_PORT="${DR_ANMAR_REMOTE_PORT:-2360}"
STATE_DIR="/tmp/dr-anmar-webcam-${UID}"
CONTROL_SOCKET="${STATE_DIR}/ssh-control.sock"
LOCAL_URL="http://127.0.0.1:${LOCAL_PORT}/"

usage() {
  cat <<'EOF'
Usage: ./dr_anmar_webcam.sh [start|open|status|stop]

Creates a private SSH tunnel to the Dr.Anmar hub and opens it through the
browser-trusted 127.0.0.1 origin, allowing webcam access without tailnet HTTPS.

Optional environment overrides:
  DR_ANMAR_SSH_TARGET   SSH destination (default: numi@100.98.17.98)
  DR_ANMAR_LOCAL_PORT   Mac loopback port (default: 12360)
  DR_ANMAR_REMOTE_PORT  Gilgamesh hub port (default: 2360)
  DR_ANMAR_NO_OPEN=1    Start the tunnel without opening a browser
EOF
}

validate_port() {
  case "$1" in
    ''|*[!0-9]*)
      printf 'Invalid TCP port: %s\n' "$1" >&2
      exit 2
      ;;
  esac
  if (( "$1" < 1 || "$1" > 65535 )); then
    printf 'TCP port is outside 1..65535: %s\n' "$1" >&2
    exit 2
  fi
}

tunnel_is_running() {
  [[ -S "$CONTROL_SOCKET" ]] &&
    ssh -S "$CONTROL_SOCKET" -O check "$SSH_TARGET" >/dev/null 2>&1
}

hub_is_ready() {
  curl --fail --silent --show-error --max-time 2 \
    --output /dev/null "${LOCAL_URL}api/status"
}

open_browser() {
  if [[ "${DR_ANMAR_NO_OPEN:-0}" != "1" ]]; then
    open "$LOCAL_URL"
  fi
}

start_tunnel() {
  if tunnel_is_running; then
    if hub_is_ready; then
      printf 'Dr.Anmar webcam tunnel is already ready at %s\n' "$LOCAL_URL"
      return
    fi
    printf 'The SSH tunnel exists, but the Dr.Anmar hub is not responding on remote port %s.\n' \
      "$REMOTE_PORT" >&2
    exit 1
  fi

  if lsof -nP -iTCP:"$LOCAL_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    printf 'Local port %s is already used by another process.\n' "$LOCAL_PORT" >&2
    printf 'Choose another with DR_ANMAR_LOCAL_PORT, for example 12361.\n' >&2
    exit 1
  fi

  mkdir -p "$STATE_DIR"
  chmod 700 "$STATE_DIR"
  if [[ -e "$CONTROL_SOCKET" ]]; then
    rm -f "$CONTROL_SOCKET"
  fi

  printf 'Connecting privately to %s...\n' "$SSH_TARGET"
  ssh -fN -M -S "$CONTROL_SOCKET" \
    -o BatchMode=yes \
    -o ConnectTimeout=10 \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=15 \
    -o ServerAliveCountMax=3 \
    -L "127.0.0.1:${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" \
    "$SSH_TARGET"

  for _attempt in 1 2 3 4 5 6 7 8 9 10; do
    if hub_is_ready; then
      printf 'Dr.Anmar webcam control is ready at %s\n' "$LOCAL_URL"
      return
    fi
    sleep 0.25
  done

  printf 'The tunnel connected, but the Dr.Anmar hub did not become ready.\n' >&2
  ssh -S "$CONTROL_SOCKET" -O exit "$SSH_TARGET" >/dev/null 2>&1 || true
  exit 1
}

stop_tunnel() {
  if tunnel_is_running; then
    ssh -S "$CONTROL_SOCKET" -O exit "$SSH_TARGET" >/dev/null
    printf 'Stopped the Dr.Anmar webcam tunnel on 127.0.0.1:%s.\n' "$LOCAL_PORT"
  else
    printf 'The Dr.Anmar webcam tunnel is not running.\n'
  fi
  if [[ -e "$CONTROL_SOCKET" ]]; then
    rm -f "$CONTROL_SOCKET"
  fi
  rmdir "$STATE_DIR" >/dev/null 2>&1 || true
}

validate_port "$LOCAL_PORT"
validate_port "$REMOTE_PORT"

case "$COMMAND" in
  start|open)
    start_tunnel
    open_browser
    ;;
  status)
    if tunnel_is_running && hub_is_ready; then
      printf 'Dr.Anmar webcam tunnel is healthy at %s\n' "$LOCAL_URL"
    elif tunnel_is_running; then
      printf 'Tunnel is connected, but the Dr.Anmar hub is unavailable.\n' >&2
      exit 1
    else
      printf 'Dr.Anmar webcam tunnel is stopped.\n'
      exit 1
    fi
    ;;
  stop)
    stop_tunnel
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
