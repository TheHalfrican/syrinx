#!/usr/bin/env bash
#
# Syrinx test runner — the local mirror of .github/workflows/ci.yml.
#
#   scripts/test.sh [rust|python|lint|audit|all]
#
# .github/workflows/ci.yml is the single source of truth: every command run
# below is copy-identical to the matching CI step. Change one, change both.
#
# Python stages use engine/.venv (this box's full engine environment) — CI
# builds a throwaway venv instead, which is the only intentional difference.
# Missing tools are reported with a pip/cargo hint; nothing is auto-installed.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV="$ROOT/engine/.venv"
VENV_PY="$VENV/bin/python"
CARGO_AUDIT="${CARGO_HOME:-$HOME/.cargo}/bin/cargo-audit"

if [[ -t 1 ]]; then
    BOLD=$'\033[1m'; RED=$'\033[31m'; GREEN=$'\033[32m'
    YELLOW=$'\033[33m'; BLUE=$'\033[34m'; RESET=$'\033[0m'
else
    BOLD=''; RED=''; GREEN=''; YELLOW=''; BLUE=''; RESET=''
fi

log()  { printf '\n%s==>%s %s%s%s\n' "$BLUE" "$RESET" "$BOLD" "$*" "$RESET"; }
hint() { printf '%shint:%s %s\n' "$YELLOW" "$RESET" "$*" >&2; }
die()  { printf '%serror:%s %s\n' "$RED" "$RESET" "$*" >&2; exit 1; }

usage() {
    cat <<'EOF'
Syrinx test runner — mirrors .github/workflows/ci.yml.

usage: scripts/test.sh [stage]

stages:
  lint     ruff check engine                      (seconds)
  python   pytest engine/tests + coverage gate    (seconds)
  rust     cargo clippy -D warnings + cargo test  (minutes)
  audit    pip-audit + cargo audit                (minutes)
  all      every stage, cheapest first  [default]
EOF
}

# --------------------------------------------------------------------------
# guards — fail loudly with an install hint instead of silently skipping
# --------------------------------------------------------------------------

need_venv() {
    [[ -x "$VENV_PY" ]] && return 0
    hint "create the engine venv first:"
    hint "  python3 -m venv engine/.venv && engine/.venv/bin/pip install -e engine"
    die "no Python interpreter at engine/.venv/bin/python"
}

# need_venv_bin <executable-in-venv-bin> <pip install args…>
need_venv_bin() {
    local bin="$1"; shift
    [[ -x "$VENV/bin/$bin" ]] && return 0
    hint "install it into the engine venv:"
    hint "  engine/.venv/bin/pip install $*"
    die "$bin is not installed in engine/.venv"
}

# need_venv_module <import name> <pip install args…>
need_venv_module() {
    local mod="$1"; shift
    "$VENV_PY" -c "import $mod" >/dev/null 2>&1 && return 0
    hint "install it into the engine venv:"
    hint "  engine/.venv/bin/pip install $*"
    die "python module '$mod' is missing from engine/.venv"
}

# --------------------------------------------------------------------------
# stages — commands mirror ci.yml exactly
# --------------------------------------------------------------------------

# ci.yml: job "python" → step "Lint (ruff)". Lint only, no formatter gate.
stage_lint() {
    need_venv
    need_venv_bin ruff "ruff"
    log "lint — ruff check engine"
    "$VENV/bin/ruff" check engine
}

# ci.yml: job "python" → step "Tests + coverage". The fail_under threshold
# lives in engine/pyproject.toml [tool.coverage.report]; --cov-config points
# coverage at it since we run from the repo root.
stage_python() {
    need_venv
    need_venv_module pytest "pytest pytest-cov"
    need_venv_module pytest_cov "pytest-cov"
    [[ -d "$ROOT/engine/tests" ]] || die "engine/tests does not exist yet — nothing to run"
    log "python — pytest engine/tests --cov=syrinx_engine"
    "$VENV_PY" -m pytest engine/tests --cov=syrinx_engine --cov-config=engine/pyproject.toml
}

# ci.yml: job "rust" → steps "Clippy (warnings are errors)" and "Tests".
stage_rust() {
    log "rust — cargo clippy --workspace --all-targets -- -D warnings"
    cargo clippy --workspace --all-targets -- -D warnings
    log "rust — cargo test --workspace"
    cargo test --workspace
}

# ci.yml: job "audit" → steps "pip-audit" and "cargo audit". CI audits a fresh
# install of the real dependency set; locally this audits engine/.venv, which
# is that same set plus whatever else this box has installed.
stage_audit() {
    need_venv
    need_venv_bin pip-audit "pip-audit"
    log "audit — pip-audit (engine/.venv)"
    # transformers pin ignore mirrors ci.yml (see the comment there)
    "$VENV/bin/pip-audit" --skip-editable --ignore-vuln PYSEC-2026-2290

    if [[ ! -x "$CARGO_AUDIT" ]]; then
        hint "install it with:"
        hint "  cargo install cargo-audit"
        die "cargo-audit not found at $CARGO_AUDIT"
    fi
    log "audit — cargo audit"
    cargo audit
}

# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------

RESULTS=()
CURRENT=''

summary() {
    local status=$?
    [[ -n "$CURRENT" ]] && RESULTS+=("$(printf '%sFAIL%s  %-6s' "$RED" "$RESET" "$CURRENT")")
    if ((${#RESULTS[@]})); then
        printf '\n%s────────────────────────────%s\n' "$BLUE" "$RESET"
        printf '  %s\n' "${RESULTS[@]}"
    fi
    exit "$status"
}
trap summary EXIT

run_stage() {
    CURRENT="$1"
    local start=$SECONDS
    # errexit aborts the script (and fires the summary trap) if the stage fails
    "stage_$1"
    RESULTS+=("$(printf '%sPASS%s  %-6s  %ss' "$GREEN" "$RESET" "$1" "$((SECONDS - start))")")
    CURRENT=''
}

case "${1:-all}" in
    # cheapest feedback first; CI runs these as parallel jobs instead
    all)                 STAGES=(lint python rust audit) ;;
    lint|python|rust|audit) STAGES=("$1") ;;
    -h|--help|help)      usage; trap - EXIT; exit 0 ;;
    *)                   usage >&2; die "unknown stage: $1" ;;
esac

for stage in "${STAGES[@]}"; do
    run_stage "$stage"
done
