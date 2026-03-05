#!/usr/bin/env bash
# Automated git pull for CVIAAR. Run from cron or systemd timer.
# Usage: ./git-pull.sh [REPO_DIR]

set -e
REPO_DIR="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_DIR"

# Fix ownership of Docker-created files so git can modify them
if [[ -d "$REPO_DIR/data/offline" ]]; then
    sudo chown -R "${SUDO_USER:-$USER}:${SUDO_GID:-$(id -gn)}" "$REPO_DIR/data/offline" 2>/dev/null || true
fi

git fetch origin

# Prefer fast-forward pull; if it fails (e.g. local changes overwritten), reset and retry
if git pull --ff-only 2>/dev/null; then
    : # success
else
    git restore --staged data/lbph_model.yml data/offline/cviaar_local.sqlite3 2>/dev/null || true
    git restore data/lbph_model.yml data/offline/cviaar_local.sqlite3 2>/dev/null || true
    git pull --ff-only || git pull
fi

# Restart web container so new code runs (app is mounted as volume, so no --build needed; avoids pip when offline)
if command -v docker >/dev/null 2>&1; then
    docker compose -f "$REPO_DIR/docker-compose.yml" --project-directory "$REPO_DIR" up -d web 2>/dev/null || true
fi
