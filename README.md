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
ollama pull qwen3:8b               # LLM de chat (com tool calling)
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
| `/health` | GET | Status, versão e estado do Ollama/modelos (503 se degradado) |
| `/chat` | POST | Resposta completa `{response, model, session_id, tool_calls}` |
| `/chat/stream` | POST | Resposta em streaming (`text/plain`) |
| `/sessions` | POST / GET | Cria uma conversa / lista as conversas (mais recentes primeiro) |
| `/sessions/{id}` | GET / DELETE | Conversa com todas as mensagens / apaga a conversa |
| `/memory` | POST | Salva fatos na memória de longo prazo `{texts, metadata?}` |
| `/memory/search` | GET | Busca semântica `?q=...&k=4&min_score=0` |
| `/memory/{id}` | DELETE | Remove uma memória |
| `/memory/stats` | GET | Total de memórias na coleção |
| `/tools` | GET | Tools carregadas (nome, descrição, argumentos) |

O corpo do chat aceita `message`, `use_memory` e `use_tools` (ambos com padrão `true`) e **uma** das formas de contexto:

- `session_id`: o Origin guarda a conversa em SQLite (`data/origin.db`) e envia ao modelo as últimas `SESSION_HISTORY_LIMIT` mensagens. As tool calls também ficam registradas.
- `history`: você mesmo envia o histórico (`[{"role": "user" \| "assistant", "content": "..."}]`), sem nada gravado no servidor.

```bash
SID=$(curl -s -X POST http://127.0.0.1:8000/sessions | jq -r .id)
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" \
  -d "{\"message\": \"Minha irmã Júlia adora chocolate amargo, guarda isso.\", \"session_id\": \"$SID\"}"
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" \
  -d "{\"message\": \"O que eu levo de presente pra ela?\", \"session_id\": \"$SID\"}"
```

### Memória de longo prazo (RAG)

A cada mensagem, o Origin busca na memória vetorial local (Chroma em `data/.chroma`) os fatos mais relevantes e os injeta no prompt de sistema. Só entram fatos com similaridade de cosseno ≥ `MEMORY_MIN_SCORE` (padrão `0.45`), no máximo `MEMORY_TOP_K` (padrão `4`).

Perguntas de seguimento como "E do que **ela** gosta?" não dizem sozinhas de quem se trata. Por isso, com `MEMORY_CONTEXTUAL_RECALL=true` (padrão), a memória também é consultada com a última troca da conversa antes da pergunta, e cada fato fica com o melhor score entre as duas buscas. Num eval com 28 memórias, sendo 25 de distração, as perguntas de seguimento passaram de 5/15 para 15/15 acertos.

```bash
curl -X POST http://127.0.0.1:8000/memory \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Meu cachorro se chama Thor."], "metadata": {"source": "manual"}}'

curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Como se chama meu cachorro?"}'
```

> O modelo de embeddings padrão é o `bge-m3` porque é multilíngue. Em testes com textos em português, o `nomic-embed-text` não separava fatos relevantes de irrelevantes. Se trocar de modelo, recalibre o `MEMORY_MIN_SCORE` e recrie a coleção, porque vetores de modelos diferentes não são compatíveis.

### Tools (agente)

O modelo decide sozinho quando chamar uma tool. O Origin executa a chamada, devolve o resultado ao modelo e repete o ciclo até ele responder, com no máximo `AGENT_MAX_TOOL_ITERATIONS` chamadas. Se uma tool falhar, o erro volta para o modelo em vez de derrubar a requisição.

| Tool | O que faz |
|------|-----------|
| `remember` | Salva um fato sobre você na memória de longo prazo ("Lembre que...") |
| `get_current_datetime` | Data, hora e dia da semana locais, no idioma de `ORIGIN_LOCALE` |
| `days_until` | Dias até uma data (`MM-DD` = próxima ocorrência, ou `YYYY-MM-DD`) |

**Criando uma tool:** crie um módulo em `origin/integrations/tools/` que exponha `get_tools(ctx)`. Ele é descoberto automaticamente na inicialização, sem nenhum registro manual.

```python
# origin/integrations/tools/weather.py
from langchain_core.tools import BaseTool, tool


def get_tools(ctx) -> list[BaseTool]:  # ctx.settings, ctx.memory
    @tool
    def get_weather(city: str) -> str:
        """Return the current weather for a city the user asked about."""
        return "sunny"

    return [get_weather]
```

A docstring é o que o modelo lê para decidir quando usar a tool, então diga claramente **quando** usar e **quando não** usar. Para desativar tools sem apagar código, use `TOOLS_DISABLED='["remember"]'`, ou `TOOLS_ENABLED=false` para desligar todas.

> **Por que qwen3:8b?** Num benchmark com 9 mensagens em português, o `qwen3:8b` acertou 9/9 as decisões de usar ou não uma tool. O `llama3.1` acertou 4/9, porque chamava tools até para "Oi, tudo bem?".

**Etapa de roteamento (`AGENT_TOOL_ROUTING`).** Modelos pequenos param de chamar tools assim que começam a escrever texto. Em perguntas compostas ("Qual a capital da França **e** que dia é hoje?"), eles respondiam a parte fácil e inventavam ou pulavam a outra. Por isso, antes de responder, o agente faz um turno dedicado em que o modelo só pode chamar tools ou dizer `NONE`. Com os mesmos modelos:

| Configuração | Simples | Compostas | Latência mediana |
|---|---|---|---|
| Sem roteamento | 35/35 | 25/40 | 1,3 s |
| `OLLAMA_REASONING=true` (thinking) | — | 12/15 | 15,9 s |
| **Roteamento (padrão)** | **40/40** | **40/40** | **1,8 s** |

### Avaliação do agente

`scripts/eval_agent.py` roda casos reais contra os modelos locais, cada um com uma memória isolada. Os grupos são: `single` (uma intenção), `compound` (várias intenções), `session` (não repetir ações de mensagens anteriores e não trocar de idioma) e `followup` (recuperar fatos pelo contexto). Use-o para comparar modelos, prompts e configurações antes de mudar um padrão:

```bash
python scripts/eval_agent.py                          # configuração atual, 3 runs por caso
python scripts/eval_agent.py --runs 5 --only compound
python scripts/eval_agent.py --model llama3.1 --routing false -v
```

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
- [x] Sistema de tools plugáveis
- [ ] Agente de automação de código
- [ ] CLI / interface web

## Contribuindo

PRs são bem-vindos. Use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `chore:`, ...) e garanta `pytest` e `ruff check .` passando.

## Licença

Copyright 2026 Douglas Nicolau Norberto. Distribuído sob a licença [Apache 2.0](LICENSE), que inclui concessão explícita de licença de patentes. Veja também [NOTICE](NOTICE).
