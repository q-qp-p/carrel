#!/usr/bin/env bash
# Optional Carrel Compose shortcut. Docker + Compose v2 are the only runtime prerequisites.
# ./deploy.sh [up] [--light|--standard] [--cpu|--gpu] [--build] [--with-pdf-images]
# ./deploy.sh config|detect|status|down
# Docker Compose can also be used directly; see deployment/compose/README.md.
# Model endpoints are configured in the console. This script installs no host services.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_DIR="$ROOT/deployment/compose"
COMPOSE_ENV="$COMPOSE_DIR/.env"
CMD=up; EDITION=""; FLAVOR=""; BUILD=0; PDF=0
for arg in "$@"; do
  case "$arg" in
    up|config|detect|status|down) CMD="$arg" ;;
    --light|--standard) value="${arg#--}"; [[ -z "$EDITION" || "$EDITION" == "$value" ]] || { echo 'Choose one edition' >&2; exit 2; }; EDITION="$value" ;;
    --cpu) FLAVOR=cpu ;;
    --gpu) FLAVOR=gpu ;;
    --build) BUILD=1 ;;
    --with-pdf-images) PDF=1; BUILD=1 ;;
    --yes|-y) : ;;
    --help|-h) sed -n '2,6s/^# //p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done
get_kv() { [[ -f "$1" ]] && awk -v k="$2" 'index($0,k "=")==1 {print substr($0,length(k)+2);exit}' "$1" || true; }
set_kv() {
  local file="$1" key="$2" value="$3" tmp
  tmp="$(mktemp)"
  awk -v k="$key" -v v="$value" 'BEGIN{done=0} index($0,k "=")==1{print k "=" v;done=1;next} {print} END{if(!done)print k "=" v}' "$file" > "$tmp"
  cat "$tmp" > "$file"; rm -f "$tmp"
}
compose() { (cd "$COMPOSE_DIR" && docker compose "$@"); }
configure() {
  local saved data password profiles
  data="$(get_kv "$COMPOSE_ENV" CARREL_DATA_PATH)"; data="${data:-../../data}"
  [[ "$data" == /* ]] || data="$COMPOSE_DIR/$data"
  saved="$(cat "$data/runtime/deployment-mode" 2>/dev/null || get_kv "$COMPOSE_ENV" CARREL_EDITION)"
  if [[ -n "$saved" && "$saved" != light && "$saved" != standard ]]; then echo 'Invalid saved edition' >&2; exit 2; fi
  if [[ -n "$saved" && -n "$EDITION" && "$saved" != "$EDITION" ]]; then echo 'Edition is fixed; use a separate data directory' >&2; exit 2; fi
  EDITION="${EDITION:-${saved:-standard}}"
  if [[ ! -f "$COMPOSE_ENV" ]]; then cp "$COMPOSE_DIR/.env.example" "$COMPOSE_ENV"; chmod 600 "$COMPOSE_ENV"; fi
  set_kv "$COMPOSE_ENV" CARREL_EDITION "$EDITION"
  profiles=""; [[ "$EDITION" == standard ]] && profiles=standard
  set_kv "$COMPOSE_ENV" COMPOSE_PROFILES "$profiles"
  FLAVOR="${FLAVOR:-$(get_kv "$COMPOSE_ENV" MINERU_FLAVOR)}"; FLAVOR="${FLAVOR:-cpu}"
  if [[ "$FLAVOR" == gpu ]]; then
    set_kv "$COMPOSE_ENV" COMPOSE_FILE docker-compose.yml:compose.gpu.yml
    set_kv "$COMPOSE_ENV" MINERU_BASE_IMAGE vllm/vllm-openai:v0.21.0
  else
    set_kv "$COMPOSE_ENV" COMPOSE_FILE docker-compose.yml
    set_kv "$COMPOSE_ENV" MINERU_BASE_IMAGE python:3.12-slim
  fi
  set_kv "$COMPOSE_ENV" MINERU_FLAVOR "$FLAVOR"
  (( PDF == 0 )) || set_kv "$COMPOSE_ENV" WITH_PDF_IMAGES 1
  password="$(get_kv "$COMPOSE_ENV" NEO4J_PASSWORD)"
  if [[ -z "$password" || "$password" == CHANGE_ME ]]; then
    password="$(LC_ALL=C tr -dc 'A-Za-z0-9' </dev/urandom 2>/dev/null | head -c 32 || true)"
    [[ ${#password} -eq 32 ]] || { echo 'Could not generate database password' >&2; exit 1; }
    set_kv "$COMPOSE_ENV" NEO4J_PASSWORD "$password"
  fi
  mkdir -p "$data"
  echo "Carrel $EDITION; MinerU $FLAVOR; data: $data"
}
case "$CMD" in
  config) configure ;;
  detect) printf 'Host: %s %s\n' "$(uname -s)" "$(uname -m)"; command -v nvidia-smi >/dev/null && nvidia-smi -L || true ;;
  up)
    docker compose version >/dev/null
    configure
    if (( BUILD )); then compose build app mineru; fi
    compose up -d
    echo 'Console: http://127.0.0.1:9800 (WEB_PORT/APP_BIND_ADDRESS may override)'
    echo 'Configure model endpoints in System settings. See deployment/compose/README.md.' ;;
  status) compose ps ;;
  down) compose down; echo 'Containers stopped; data and volumes retained.' ;;
esac
