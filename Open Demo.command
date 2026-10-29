#!/bin/zsh
set -eu
PROJECT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$PROJECT_DIR"
if [[ ! -x .venv/bin/python ]]; then
  print 'The Python environment is missing. Run Setup Mac.command first.'
  read '?Press Return to close.'
  exit 1
fi
exec .venv/bin/python launch_demo.py
