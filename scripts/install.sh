#!/usr/bin/env sh
# Installs Origin on Linux / macOS: a Python virtual environment, the dependencies, a .env
# file and the Ollama models.
#
#   sh scripts/install.sh               # everything
#   sh scripts/install.sh --skip-models # download the models later (first-run screen)
set -eu
cd "$(dirname "$0")/.."

step() { printf '\n==> %s\n' "$1"; }
setting() { # value of NAME in .env, else the default
  value=$(sed -n "s/^[[:space:]]*$1[[:space:]]*=[[:space:]]*//p" .env 2>/dev/null | head -n 1)
  printf '%s' "${value:-$2}"
}

step "Python 3.11+"
PYTHON=$(command -v python3 || command -v python || true)
[ -n "$PYTHON" ] || { echo "Python not found: install Python 3.11 or newer." >&2; exit 1; }
"$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 11))' ||
  { echo "Origin needs Python 3.11 or newer ($("$PYTHON" --version) found)." >&2; exit 1; }
"$PYTHON" --version

step "Virtual environment and dependencies"
[ -d .venv ] || "$PYTHON" -m venv .venv
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -e .

step "Configuration"
if [ -f .env ]; then
  echo ".env already exists: kept as is."
else
  cp .env.example .env
  echo "Created .env from .env.example (edit it to change models, language, etc.)."
fi

step "Ollama"
if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama not found. Install it from https://ollama.com/download; the first-run screen will then download the models."
elif [ "${1:-}" = "--skip-models" ]; then
  echo "Skipping model downloads (--skip-models)."
else
  for model in "$(setting OLLAMA_MODEL qwen3:8b)" "$(setting OLLAMA_EMBED_MODEL bge-m3)"; do
    echo "ollama pull $model"
    ollama pull "$model"
  done
fi

step "Done"
echo "Start it with:  .venv/bin/python main.py"
echo "Then open:      http://127.0.0.1:$(setting ORIGIN_PORT 8000)"
echo "White label:    cp brand/brand.example.json brand/brand.json, then edit it."
