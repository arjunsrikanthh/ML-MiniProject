#!/bin/zsh
set -eu
PROJECT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$PROJECT_DIR"
if [[ ! -f /opt/homebrew/opt/libomp/lib/libomp.dylib ]]; then
  HOMEBREW_NO_AUTO_UPDATE=1 brew install libomp
fi
if [[ ! -x .venv/bin/python ]]; then
  python3.12 -m venv .venv
fi
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --prefix "$PROJECT_DIR/.venv" \
  --name ml-miniproject-mac --display-name 'ML MiniProject (Mac)'
print 'Setup complete. Double-click Open Demo.command.'
