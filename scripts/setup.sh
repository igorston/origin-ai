#!/usr/bin/env bash
# Origin — bootstrap: estrutura, git init e commit inicial.
# Uso: bash scripts/setup.sh   (a partir da raiz do repositório)
set -euo pipefail

REMOTE_URL="https://github.com/igorston/origin-ai.git"

# 1. Diretórios
mkdir -p \
  origin/{api/routes,config,core,integrations/{connectors,tools},memory/{storage,vectorstore},prompts/templates} \
  tests/{unit,integration} \
  data scripts

# 2. Arquivos (touch não sobrescreve os existentes)
touch \
  origin/__init__.py \
  origin/api/__init__.py origin/api/routes/__init__.py \
  origin/config/__init__.py origin/config/settings.py \
  origin/core/__init__.py origin/core/llm.py origin/core/agent.py \
  origin/integrations/__init__.py \
  origin/integrations/connectors/__init__.py \
  origin/integrations/tools/__init__.py \
  origin/memory/__init__.py \
  origin/memory/storage/__init__.py \
  origin/memory/vectorstore/__init__.py \
  origin/prompts/__init__.py origin/prompts/templates/system.md \
  tests/__init__.py tests/conftest.py \
  tests/unit/__init__.py tests/integration/__init__.py \
  data/.gitkeep

for f in README.md LICENSE NOTICE .gitignore main.py pyproject.toml .env.example; do
  [[ -f "$f" ]] || { echo "ERRO: $f ausente — salve os arquivos base antes." >&2; exit 1; }
done

# 3. Git
[[ -d .git ]] || git init -b main
git remote get-url origin >/dev/null 2>&1 || git remote add origin "$REMOTE_URL"
git add .
git commit -m "chore: initial commit - origin ai base structure"

echo "✔ Origin inicializado. Para publicar: git push -u origin main"
