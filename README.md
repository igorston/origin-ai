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
| `/chat/stream` | POST | Resposta em streaming (`text/plain`), só o texto |
| `/chat/events` | POST | Streaming em SSE com eventos `tool_call`, `token`, `done` e `error` |
| `/sessions` | POST / GET | Cria uma conversa / lista as conversas (mais recentes primeiro) |
| `/sessions/{id}` | GET / DELETE | Conversa com todas as mensagens / apaga a conversa |
| `/memory` | POST | Salva fatos `{texts, metadata?}` (quase-duplicatas não são salvas de novo) |
| `/memory` | GET | Lista todas as memórias, as mais novas primeiro (`?limit=&offset=`) |
| `/memory/search` | GET | Busca semântica `?q=...&k=4&min_score=0` (ignora arquivadas) |
| `/memory/{id}` | DELETE | Remove uma memória |
| `/memory/{id}/restore` | POST | Restaura uma memória arquivada |
| `/memory/stats` | GET | Total, ativas e arquivadas |
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

### Streaming com eventos (SSE)

O `/chat/stream` é prático no `curl`, mas entrega só texto. Uma interface que precise mostrar o que o agente está fazendo deve usar o `/chat/events`, que aceita o mesmo corpo:

```bash
curl -N -X POST http://127.0.0.1:8000/chat/events -H "Content-Type: application/json" \
  -d '{"message": "Lembra que eu moro em Recife e me diz quantos dias faltam pro Natal"}'
```

```text
event: tool_call
data: {"name": "remember", "args": {"fact": "Moro em Recife."}, "output": "Saved to long-term memory (id=...)."}

event: tool_call
data: {"name": "days_until", "args": {"target_date": "12-25"}, "output": "days: 90\n..."}

event: token
data: {"text": "Anotei"}

event: done
data: {"response": "Anotei que você mora em Recife. Faltam 90 dias para o Natal.", "model": "qwen3:8b", "session_id": null, "tool_calls": [...]}
```

Falhas antes do primeiro evento, como o Ollama fora do ar, viram erros HTTP (503/502) nos dois endpoints de streaming. Falhas depois que o stream começou chegam como `event: error`. Com `session_id`, a troca é gravada ao final, inclusive se o cliente desconectar no meio.

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

**Duplicatas e fatos desatualizados.** Ao salvar um fato, o Origin o compara com o que já existe (limiares calibrados com o bge-m3):

| Similaridade | O que acontece |
|---|---|
| ≥ `MEMORY_DEDUP_THRESHOLD` (0,92) | É o mesmo fato: não salva de novo |
| ≥ `MEMORY_CONFLICT_THRESHOLD` (0,72) | Uma pergunta de sim/não ao modelo (temperatura 0, ~0,1 s) decide se o fato novo substitui o antigo. "Meu time é o Náutico" substitui "Meu time é o Sport"; "Minha filha se chama Laura" **não** substitui "Meu filho se chama Pedro" |
| abaixo | Fatos independentes |

Fatos substituídos são **arquivados**, não apagados: saem da busca, mas continuam em `GET /memory` e podem ser restaurados com `POST /memory/{id}/restore`. A pergunta foi escrita para errar para o lado seguro. Num teste com 19 pares, ela não teve nenhum falso positivo (nunca substituiu um fato ainda verdadeiro), mas deixou passar 3 substituições reais. Para esquecer algo de propósito ("esquece aquilo da alergia"), existe a tool `forget`, que apaga de verdade.

> O modelo de embeddings padrão é o `bge-m3` porque é multilíngue. Em testes com textos em português, o `nomic-embed-text` não separava fatos relevantes de irrelevantes. Se trocar de modelo, recalibre o `MEMORY_MIN_SCORE` e recrie a coleção, porque vetores de modelos diferentes não são compatíveis.

### Tools (agente)

O modelo decide sozinho quando chamar uma tool. O Origin executa a chamada, devolve o resultado ao modelo e repete o ciclo até ele responder, com no máximo `AGENT_MAX_TOOL_ITERATIONS` chamadas. Se uma tool falhar, o erro volta para o modelo em vez de derrubar a requisição.

| Tool | O que faz |
|------|-----------|
| `remember` | Salva um fato sobre você ("Lembre que...", "Mudei de..."), sem duplicar e arquivando o que ficou desatualizado |
| `forget` | Apaga uma memória por id ou descrição ("Esquece aquilo da...") |
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

A etapa de roteamento roda com temperatura 0 (`AGENT_ROUTING_TEMPERATURE`), porque decidir quais tools chamar é uma classificação e a aleatoriedade ali só faz o modelo pular tools. Com 10 runs das perguntas compostas, o resultado foi 78/80 com 0,7 e 80/80 com 0.

**Verificação de afirmações.** Às vezes o modelo diz "anotei!" sem ter chamado o `remember`. Se a resposta final afirma um efeito (salvar, apagar) cuja tool não rodou naquele turno, o agente faz um turno curto de verificação: o modelo chama a tool agora, para a afirmação virar verdade, ou responde `NONE`. O turno extra só acontece quando uma dessas afirmações aparece.

**Idioma.** As saídas das tools são em inglês e são a última coisa que o modelo lê antes de responder, o que fazia modelos pequenos responderem em inglês. Por isso, o agente detecta o idioma da mensagem (português, inglês ou espanhol, por palavras comuns) e anexa à saída da tool uma instrução **escrita nesse idioma** ("Responda ao usuário em português..."). Duas tentativas anteriores falharam: citar a mensagem do usuário fazia o modelo repeti-la como resposta, e uma instrução em inglês ("reply in Portuguese") ainda puxava palavras em inglês. Limitação conhecida: em cerca de 1 a cada 35 respostas após uma tool, ainda escapa uma palavra em inglês ("algo else").

### Avaliação do agente

`scripts/eval_agent.py` roda casos reais contra os modelos locais, cada um com uma memória isolada. Os grupos são: `single` (uma intenção), `compound` (várias intenções), `session` (não repetir ações de mensagens anteriores e não trocar de idioma), `memory` (duplicatas, substituições e esquecimento) e `followup` (recuperar fatos pelo contexto). Use-o para comparar modelos, prompts e configurações antes de mudar um padrão:

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
