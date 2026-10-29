#!/bin/zsh
set -eu
PROJECT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$PROJECT_DIR"
exec .venv/bin/python launch_live_demo.py
