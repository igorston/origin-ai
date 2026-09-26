<div align="center">

# 🧬 Origin

**A fundação open source para assistentes de IA pessoais — 100% local, modular e sua.**

[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-000000?logo=ollama&logoColor=white)](https://ollama.com/)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-D7FF64?logo=ruff&logoColor=black)](https://github.com/astral-sh/ruff)
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
ollama pull nomic-embed-text       # embeddings

# 5. Configuração
cp .env.example .env

# 6. Execute
python main.py
```

A API sobe em `http://127.0.0.1:8000` — docs interativas em `/docs` e healthcheck em `/health`.

```bash
pytest          # testes
ruff check .    # lint
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
- [ ] Engine LLM com Ollama + streaming
- [ ] Memória vetorial local (Chroma)
- [ ] Sistema de tools plugáveis
- [ ] Agente de automação de código
- [ ] CLI / interface web

## Contribuindo

PRs são bem-vindos. Use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `chore:`, ...) e garanta `pytest` e `ruff check .` passando.
