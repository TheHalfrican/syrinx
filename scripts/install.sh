#!/usr/bin/env bash
#
# Syrinx desktop install (beta) — makes this checkout behave like an installed
# app: launcher entry, release binaries on PATH, engine auto-started on demand
# by D-Bus activation under systemd --user.
#
#   scripts/install.sh              install
#   scripts/install.sh --uninstall  reverse everything
#
# BETA COUPLING, ON PURPOSE: the engine keeps running FROM THIS CHECKOUT. Its
# venvs (engine/.venv, .venv-seedvc, .venv-vevo) are machine-local and its
# worker paths resolve relative to the engine source dir, so ExecStart points
# here rather than at /usr/bin. Move or delete this clone and the engine stops
# resolving — re-run this script after relocating it.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BIN_DIR="$HOME/.local/bin"
UNIT_DIR="$HOME/.config/systemd/user"
DBUS_DIR="$HOME/.local/share/dbus-1/services"
APPS_DIR="$HOME/.local/share/applications"
ICON_DIR="$HOME/.local/share/icons/hicolor/scalable/apps"

BINARIES=(syrinx-app syrinx-dictate syrinx-dictate-pill)

if [[ -t 1 ]]; then
    BOLD=$'\033[1m'; RED=$'\033[31m'; GREEN=$'\033[32m'
    YELLOW=$'\033[33m'; BLUE=$'\033[34m'; RESET=$'\033[0m'
else
    BOLD=''; RED=''; GREEN=''; YELLOW=''; BLUE=''; RESET=''
fi

log()  { printf '\n%s==>%s %s%s%s\n' "$BLUE" "$RESET" "$BOLD" "$*" "$RESET"; }
ok()   { printf '%s  ok%s %s\n' "$GREEN" "$RESET" "$*"; }
hint() { printf '%shint:%s %s\n' "$YELLOW" "$RESET" "$*" >&2; }
die()  { printf '%serror:%s %s\n' "$RED" "$RESET" "$*" >&2; exit 1; }

# Both optional; a box without them just doesn't get a cache bump.
refresh_caches() {
    if command -v update-desktop-database >/dev/null; then
        update-desktop-database "$APPS_DIR" || true
    fi
    if command -v gtk-update-icon-cache >/dev/null; then
        gtk-update-icon-cache -qtf "$HOME/.local/share/icons/hicolor" || true
    fi
}

# Render @REPO@/@BIN@ templates. Paths go through sed's s||| with | as the
# delimiter — fine for absolute paths, which never contain a pipe.
render() {
    local src="$1" dst="$2"
    mkdir -p "$(dirname "$dst")"
    sed -e "s|@REPO@|$ROOT|g" -e "s|@BIN@|$BIN_DIR|g" "$src" > "$dst"
    ok "$dst"
}

# --------------------------------------------------------------------------
# uninstall

uninstall() {
    log "Stopping the engine"
    # The unit may not exist (partial install); never let that abort us.
    systemctl --user stop syrinx-engine.service 2>/dev/null || true
    systemctl --user disable syrinx-engine.service 2>/dev/null || true

    log "Removing installed files"
    local f
    for f in "${BINARIES[@]/#/$BIN_DIR/}" \
             "$UNIT_DIR/syrinx-engine.service" \
             "$DBUS_DIR/sh.syrinx.Engine.service" \
             "$APPS_DIR/syrinx.desktop" \
             "$ICON_DIR/syrinx.svg"; do
        if [[ -e "$f" ]]; then
            rm -f "$f"
            ok "removed $f"
        fi
    done

    log "Refreshing caches"
    systemctl --user daemon-reload
    refresh_caches

    printf '\n%sSyrinx uninstalled.%s The checkout at %s is untouched.\n' \
        "$BOLD" "$RESET" "$ROOT"
    exit 0
}

case "${1-}" in
    "")          ;;
    --uninstall) uninstall ;;
    -h|--help)
        cat <<'EOF'
Syrinx desktop install (beta).

usage: scripts/install.sh [--uninstall]

  (no args)     build release binaries, install them plus the systemd unit,
                D-Bus activation file, desktop entry and icon
  --uninstall   stop the engine and remove every installed file

The engine runs from this checkout by design — see the header comment.
EOF
        exit 0 ;;
    *)           die "unknown argument: $1 (try --uninstall)" ;;
esac

# --------------------------------------------------------------------------
# guards

[[ -x "$ROOT/engine/.venv/bin/syrinx-engine" ]] || die \
    "engine/.venv/bin/syrinx-engine not found — run the engine setup first:
       cd engine && python -m venv .venv && .venv/bin/pip install -e ."

command -v cargo >/dev/null || die "cargo not found — install rust first"

# --------------------------------------------------------------------------
# build + install

log "Building release binaries (this takes a few minutes)"
cargo build --release
ok "target/release"

log "Installing binaries to $BIN_DIR"
mkdir -p "$BIN_DIR"
for b in "${BINARIES[@]}"; do
    [[ -f "$ROOT/target/release/$b" ]] || die "target/release/$b missing after build"
    install -Dm755 "$ROOT/target/release/$b" "$BIN_DIR/$b"
    ok "$BIN_DIR/$b"
done

log "Installing service + desktop files"
render "$ROOT/packaging/syrinx-engine.service.in"   "$UNIT_DIR/syrinx-engine.service"
render "$ROOT/packaging/sh.syrinx.Engine.service.in" "$DBUS_DIR/sh.syrinx.Engine.service"
render "$ROOT/packaging/syrinx.desktop.in"          "$APPS_DIR/syrinx.desktop"
install -Dm644 "$ROOT/packaging/syrinx.svg" "$ICON_DIR/syrinx.svg"
ok "$ICON_DIR/syrinx.svg"

log "Refreshing caches"
systemctl --user daemon-reload
ok "systemctl --user daemon-reload"
# dbus-daemon only rescans its service dirs when asked; without this the very
# first activation of a fresh install fails with "The name is not activatable".
busctl --user call org.freedesktop.DBus /org/freedesktop/DBus \
    org.freedesktop.DBus ReloadConfig >/dev/null 2>&1 \
    && ok "dbus ReloadConfig" || hint "couldn't reach the session bus — log out and back in before first launch"
refresh_caches

# --------------------------------------------------------------------------
# summary

cat <<EOF

${BOLD}Syrinx is installed.${RESET}

  Launch          from your app launcher ("Syrinx"), or: syrinx-app
  Dictation       syrinx-dictate toggle   (bind it — see packaging/hyprland.conf)
  Engine logs     journalctl --user -u syrinx-engine -f

The engine is ${BOLD}not${RESET} enabled at login, and doesn't need to be — D-Bus
activation starts it the moment the app or the pill first talks to
sh.syrinx.Engine, and systemd restarts it on failure. First call after a cold
start waits ~15s for model warmup; every call after that is instant.

${YELLOW}Beta caveat:${RESET} the engine runs from this checkout
  $ROOT
because its venvs and worker paths are machine-local. Don't move or delete it —
if you do, re-run this script from the new location.

Uninstall with: scripts/install.sh --uninstall
EOF
