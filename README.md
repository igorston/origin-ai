<div align="center">

# 🧬 Origin

**A fundação open source para assistentes de IA pessoais — 100% local, modular e sua.**

[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-000000?logo=ollama&logoColor=white)](https://ollama.com/)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-D7FF64?logo=ruff&logoColor=black)](https://github.com/astral-sh/ruff)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-experimental-orange)](#roadmap)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/igorston/origin-ai/pulls)

</div>

---

## Por que Origin?

Seus dados, seus modelos, sua máquina. O **Origin** é um núcleo de IA que roda inteiramente offline sobre LLMs locais (via Ollama), sem enviar um byte para a nuvem. Ele foi pensado como **sandbox de experimentação** e **fundação modular** para:

- 🤖 **Assistentes pessoais** com memória de longo prazo local (vetores + banco).
- ⚡ **Vibe coding** — automação de código assistida por agentes e ferramentas.
- 🔌 **Integrações de sistemas** — conecte APIs, arquivos e serviços como *tools* plugáveis.
- 🧪 **Experimentação** — troque modelos, prompts e estratégias de memória sem reescrever o core.

## Pré-requisitos

| Requisito | Versão | Observação |
|-----------|--------|------------|
| [Python](https://www.python.org/downloads/) | 3.11+ | |
| [Ollama](https://ollama.com/download) | latest | Servidor de LLM local |
| [Git](https://git-scm.com/) | 2.40+ | |
| RAM | 8 GB+ | 16 GB+ recomendado para modelos 7B–8B |
| GPU (opcional) | — | NVIDIA (CUDA) / Apple Silicon (Metal) acelera a inferência |

## Quickstart

```bash
# 1. Clone
git clone https://github.com/igorston/origin-ai.git
cd origin-ai

# 2. Ambiente virtual
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Dependências
pip install -e ".[dev]"

# 4. Modelos locais
ollama pull llama3.1               # LLM de chat
ollama pull bge-m3                 # embeddings multilíngues (memória)

# 5. Configuração
cp .env.example .env

# 6. Execute
python main.py
```

A API sobe em `http://127.0.0.1:8000` — docs interativas em `/docs` e healthcheck em `/health`.

```bash
# Resposta completa
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Quem é você?"}'

# Streaming (token a token)
curl -N -X POST http://127.0.0.1:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Explique RAG em 3 frases.", "history": []}'
```

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/health` | GET | Status e versão |
| `/chat` | POST | Resposta completa `{response, model}` |
| `/chat/stream` | POST | Resposta em streaming (`text/plain`) |
| `/memory` | POST | Salva fatos na memória de longo prazo `{texts, metadata?}` |
| `/memory/search` | GET | Busca semântica `?q=...&k=4&min_score=0` |
| `/memory/{id}` | DELETE | Remove uma memória |
| `/memory/stats` | GET | Total de memórias na coleção |

O corpo do chat aceita `message`, um `history` opcional (`[{"role": "user" \| "assistant", "content": "..."}]`) e `use_memory` (padrão `true`).

### Memória de longo prazo (RAG)

A cada mensagem, o Origin busca na memória vetorial local (Chroma em `data/.chroma`) os fatos mais relevantes e os injeta no prompt de sistema. Só entram fatos com similaridade de cosseno ≥ `MEMORY_MIN_SCORE` (padrão `0.45`), no máximo `MEMORY_TOP_K` (padrão `4`).

```bash
curl -X POST http://127.0.0.1:8000/memory \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Meu cachorro se chama Thor."], "metadata": {"source": "manual"}}'

curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Como se chama meu cachorro?"}'
```

> O modelo de embeddings padrão é o `bge-m3` porque é multilíngue. Em testes com textos em português, o `nomic-embed-text` não separava fatos relevantes de irrelevantes. Se trocar de modelo, recalibre o `MEMORY_MIN_SCORE` e recrie a coleção, porque vetores de modelos diferentes não são compatíveis.

```bash
pytest                  # todos os testes (integração é pulada se o Ollama estiver offline)
pytest -m "not integration"
ruff check . && ruff format --check .
```

## Arquitetura

```mermaid
flowchart LR
    Client([CLI / UI / Webhooks]) --> API[FastAPI<br/>origin/api]
    API --> Core[Core Engine<br/>origin/core]
    Core --> LLM[(Ollama<br/>LLM local)]
    Core --> Prompts[Prompts<br/>origin/prompts]
    Core --> Memory[Memória<br/>origin/memory]
    Core --> Integrations[Integrações & Tools<br/>origin/integrations]
    Memory --> Vec[(Vector Store<br/>Chroma/Qdrant)]
    Memory --> DB[(SQLite)]
    Integrations --> Ext[APIs · Arquivos · Shell · Sistemas]
```

| Camada | Diretório | Responsabilidade |
|--------|-----------|------------------|
| **API** | `origin/api/` | Rotas HTTP, schemas, streaming. Camada fina, sem regra de negócio. |
| **Core** | `origin/core/` | Engine LLM (LangChain + Ollama), orquestração de agentes e chains. |
| **Prompts** | `origin/prompts/` | Templates versionados, desacoplados do código. |
| **Memória** | `origin/memory/` | Memória de curto prazo (conversa) e longo prazo (vetores + SQLite). |
| **Integrações** | `origin/integrations/` | *Tools* e *connectors* plugáveis (APIs, filesystem, git, etc.). |
| **Config** | `origin/config/` | Settings tipados via variáveis de ambiente. |

```
origin-ai/
├── main.py                  # Entry point
├── origin/
│   ├── api/routes/          # Endpoints FastAPI
│   ├── config/              # Settings (pydantic-settings)
│   ├── core/                # LLM engine, agentes, chains
│   ├── integrations/
│   │   ├── connectors/      # Clientes de sistemas externos
│   │   └── tools/           # Tools LangChain
│   ├── memory/
│   │   ├── storage/         # SQLite / histórico
│   │   └── vectorstore/     # Chroma / Qdrant
│   └── prompts/templates/   # Prompts em Markdown
├── data/                    # Dados locais (ignorado pelo git)
├── scripts/                 # Automação / setup
└── tests/{unit,integration}/
```

### Princípios

1. **Local-first** — nenhuma dependência de nuvem obrigatória.
2. **Plugável** — cada integração é um módulo isolado com interface comum.
3. **Core agnóstico** — trocar de modelo ou vector store é configuração, não refatoração.
4. **Prompts como dados** — versionados e testáveis.

## Roadmap

- [x] Estrutura base e entry point
- [x] Engine LLM com Ollama + streaming
- [x] Memória vetorial local (Chroma)
- [ ] Sistema de tools plugáveis
- [ ] Agente de automação de código
- [ ] CLI / interface web

## Contribuindo

PRs são bem-vindos. Use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `chore:`, ...) e garanta `pytest` e `ruff check .` passando.

## Licença

Copyright 2026 Douglas Nicolau Norberto. Distribuído sob a licença [Apache 2.0](LICENSE), que inclui concessão explícita de licença de patentes. Veja também [NOTICE](NOTICE).
