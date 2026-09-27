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

Abra **http://127.0.0.1:8000** no navegador para usar a interface web. A API continua disponível no mesmo endereço, com documentação interativa em `/docs` e healthcheck em `/health`.

### Interface web

A interface é servida pelo próprio Origin (`origin/web/static`): HTML, CSS e JavaScript puros, sem etapa de build e sem nenhum recurso de CDN, então funciona offline. Ela foi feita para testes manuais:

| Área | O que faz |
|---|---|
| **Conversas** (esquerda) | Cria, retoma e apaga sessões. O histórico fica no servidor e sobrevive a um reload. |
| **Chat** | Streaming token a token via `/chat/events`. Cada tool call aparece num cartão expansível com argumentos e resultado. O rodapé de cada resposta mostra o tempo até o 1º token e o total. O botão **Parar** interrompe a geração. |
| **🧠 Memória** (canto inferior esquerdo, com o total de memórias ativas) | Gerenciador completo: busca por significado (com score), filtros por estado (ativas, arquivadas, todas) e por origem (agente, manual). Permite **editar o texto** (clique em "editar" ou dê duplo clique; Enter salva, Esc cancela, e o fato ganha novo embedding), arquivar, restaurar, apagar e adicionar. Com **✨ Otimizar com IA** (ligado por padrão), o que você escreve passa pela curadoria descrita abaixo antes de ser gravado, e um aviso mostra o que aconteceu ("Dividida em 2 memórias", "Já existia, mesclada"). Desligue para gravar exatamente o que escreveu. Memórias arquivadas mostram qual fato as substituiu. |
| **⚙ Configurações** (canto inferior esquerdo) | **Geral:** liga ou desliga a memória e as tools (a preferência fica salva no navegador, e o rodapé avisa quando algo está desligado). **Tools:** tools carregadas e suas descrições. **Sistema:** estado do Ollama e dos modelos. |
| **Medidor de contexto** (canto superior direito do chat) | Quanto da janela do modelo a conversa ocupa (🟢 folgado, 🟡 a partir de 60%, 🔴 a partir de 90%). Clique para ver a composição: instruções e tools, resumo, mensagens recentes, memórias, reserva para a resposta, e quantas otimizações e condensações já ocorreram. Um divisor 🗜 marca onde as mensagens antigas foram resumidas (clique para ver o resumo que o modelo recebe). Perto do limite operacional aparece um aviso. Quando a conversa se encerra, o campo de mensagem é bloqueado e o botão **Continuar em nova conversa** abre uma conversa que já começa com o resumo. Veja [Contexto da conversa](#contexto-da-conversa). |
| **Indicador de status** | 🟢 online · 🟡 **carregando** (modelos fora da memória do Ollama; a próxima resposta pode levar ~15 s) · 🔴 degradado/offline. |

Tema claro e escuro seguem o sistema, e o layout se adapta ao celular.

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
| `/` | GET | Interface web |
| `/health` | GET | Status, versão, `ready` (modelos carregados) e estado do Ollama (503 se degradado) |
| `/chat` | POST | Resposta completa `{response, model, session_id, tool_calls, context, compactions, trimmed}` |
| `/chat/stream` | POST | Resposta em streaming (`text/plain`), só o texto |
| `/chat/events` | POST | Streaming em SSE com eventos `context`, `tool_call`, `token`, `done` e `error` |
| `/sessions` | POST / GET | Cria uma conversa / lista as conversas (mais recentes primeiro) |
| `/sessions/{id}` | GET / DELETE | Conversa com todas as mensagens, resumo e uso de contexto / apaga a conversa |
| `/sessions/{id}/continue` | POST | Cria uma conversa nova que começa com o resumo desta (usada quando ela se encerra) |
| `/memory` | POST | Salva fatos `{texts, metadata?}` (quase-duplicatas não são salvas de novo) |
| `/memory` | GET | Lista todas as memórias, as mais novas primeiro (`?limit=&offset=`) |
| `/memory/search` | GET | Busca semântica `?q=...&k=4&min_score=0` (ignora arquivadas) |
| `/memory/{id}` | PATCH | Edita o texto (com novo embedding) e/ou arquiva: `{content?, archived?, optimize?}` |
| `/memory/{id}` | DELETE | Remove uma memória |
| `/memory/{id}/restore` | POST | Restaura uma memória arquivada |
| `/memory/stats` | GET | Total, ativas e arquivadas |
| `/tools` | GET | Tools carregadas (nome, descrição, argumentos) |

O corpo do chat aceita `message`, `use_memory` e `use_tools` (ambos com padrão `true`) e **uma** das formas de contexto:

- `session_id`: o Origin guarda a conversa em SQLite (`data/origin.db`) e cuida da janela de contexto sozinho (ver [Contexto da conversa](#contexto-da-conversa)). As tool calls também ficam registradas.
- `history`: você mesmo envia o histórico (`[{"role": "user" \| "assistant", "content": "..."}]`), sem nada gravado no servidor. Se ele não couber na janela, as mensagens mais antigas são descartadas e `trimmed` diz quantas.

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
event: context
data: {"usage": {"used": 1410, "usable": 5120, "percent": 0.275, "state": "ok", ...}, "compactions": [], "trimmed": 0}

event: tool_call
data: {"name": "remember", "args": {"fact": "Moro em Recife."}, "output": "Saved to long-term memory (id=...)."}

event: tool_call
data: {"name": "days_until", "args": {"target_date": "12-25"}, "output": "days: 90\n..."}

event: token
data: {"text": "Anotei"}

event: done
data: {"response": "Anotei que você mora em Recife. Faltam 90 dias para o Natal.", "model": "qwen3:8b", "session_id": null, "tool_calls": [...], "context": {...}}
```

`context` abre o stream com o orçamento antes do turno e com as otimizações feitas para ele caber. O `done` traz o uso depois do turno, medido pelo Ollama.

Falhas antes do primeiro evento, como o Ollama fora do ar, viram erros HTTP (503/502) nos dois endpoints de streaming. Falhas depois que o stream começou chegam como `event: error`. Com `session_id`, a troca é gravada ao final, inclusive se o cliente desconectar no meio.

### Contexto da conversa

O modelo só enxerga `OLLAMA_NUM_CTX` tokens (6144 por padrão). Sem esse ajuste, o Ollama usava 4096 e cortava em silêncio o começo de prompts longos, inclusive as instruções. Cada turno envia as instruções e os schemas das tools (~1.230 tokens, medidos no warmup), o resumo da conversa, as mensagens recentes, as memórias recuperadas e a mensagem nova. `CONTEXT_REPLY_RESERVE` (1024) fica livre para a resposta. O restante é o orçamento **utilizável**, e é ele que o medidor da interface mostra.

**Otimização automática.** Quando o prompt passa de 75% do orçamento, as mensagens mais antigas são incorporadas a um **resumo contínuo**, até o uso voltar para menos de 50%. As 6 mais recentes continuam literais. O histórico completo continua salvo e visível. Só a visão do modelo muda: ele recebe o resumo no prompt de sistema. O resumo tem três seções:

- **USER FACTS**: fatos e pedidos do usuário (nomes, datas, números, códigos, arquivos, funções), copiados exatamente e nunca condensados.
- **TOPICS**: assuntos da conversa e as conclusões, condensados livremente.
- **PRESERVED**: uma rede de segurança no código. Se o modelo deixar de fora um detalhe que o usuário escreveu (um número, um código, `snake_case`, um nome de arquivo, um nome próprio), a frase original do usuário entra aqui **literalmente**.

As estimativas de tokens são calibradas pela medição real do turno anterior. Textos em prosa costumam pesar metade da estimativa, e códigos e preços pesam mais. Sem essa calibração, o chat compactaria cedo demais ou estouraria a reserva.

**Limite operacional.** Incorporar mensagens ao resumo é sustentável e não tem limite. O que perde informação é **condensar** o resumo quando ele passa do orçamento (`CONTEXT_SUMMARY_MAX_TOKENS`, no máximo 20% da janela). Por isso só as condensações são contadas: depois de `CONTEXT_MAX_COMPRESSIONS` (8), o resumo pode crescer além do orçamento, e a conversa **só é encerrada quando nem ele e as últimas mensagens cabem mais na janela**. A conversa encerrada fica somente leitura (409 com `{closed, reason, context}`), e `POST /sessions/{id}/continue` cria uma conversa nova que começa com o resumo. Esse é o único momento em que até os fatos do usuário podem ser condensados, se sozinhos não couberem. Uma mensagem que sozinha não cabe na janela é recusada com 413, e a conversa continua aberta.

**VRAM.** Os modelos de chat e de embeddings precisam caber juntos na GPU. Numa GPU de 8 GB, `qwen3:8b` e `bge-m3` cabem com 6144 (5,5 + 0,6 GB). Com 8192, o Ollama trocava os dois modelos a cada chamada (+4 a 5 s por chamada). O warmup avisa no log quando os modelos não couberem juntos, e o startup avisa quando a janela é pequena demais para a otimização funcionar bem.

`scripts/eval_context.py` roda uma conversa longa real com janela de 4096 tokens e memória e tools desligadas, então os fatos só podem voltar pelo resumo. Ela planta 4 mensagens com fatos (voo e código da reserva, gerente, função com bug, orçamento), faz 16 perguntas de conhecimento geral e depois pergunta pelos fatos. O resultado foi **5/5 fatos recuperados** depois de 4 otimizações, em três rodadas seguidas. Antes das seções fixas e da rede de segurança, eram 0/5, e a conversa chegava ao limite depois de ~25 mensagens.

### Memória de longo prazo (RAG)

A cada mensagem, o Origin busca na memória vetorial local (Chroma em `data/.chroma`) os fatos mais relevantes e os injeta no prompt de sistema. Só entram fatos com similaridade de cosseno ≥ `MEMORY_MIN_SCORE` (padrão `0.45`), no máximo `MEMORY_TOP_K` (padrão `4`).

Perguntas compostas diluem a busca: em "Onde eu moro **e** quantos dias faltam pro Natal?", o fato "Moro em São Paulo" pontua 0,44 (abaixo do limiar), contra 0,56 com a pergunta sozinha, e o modelo chegou a inventar outra cidade. Por isso, cada parte da mensagem também é buscada separadamente.

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
| ≥ `MEMORY_CONFLICT_THRESHOLD` (0,55), até 3 candidatos | Uma pergunta de sim/não ao modelo (temperatura 0, em paralelo, ~0,1 s) decide se o fato novo substitui o antigo. "Moro em São Paulo" substitui "Eu moro em Recife"; "Minha filha se chama Laura" **não** substitui "Meu filho se chama Pedro" |
| abaixo | Fatos independentes |

O limiar é baixo de propósito. Contradições escritas de formas diferentes pontuam pouco ("Eu moro em Recife." / "Moro em São Paulo." = 0,58), na mesma faixa de fatos só relacionados. Por isso a similaridade só pré-seleciona candidatos, e quem decide é a pergunta.

Fatos substituídos são **arquivados**, não apagados: saem da busca, mas continuam em `GET /memory` e podem ser restaurados com `POST /memory/{id}/restore`. A pergunta foi escrita para errar para o lado seguro. Em 60 julgamentos, ela teve zero falsos positivos (nunca substituiu um fato ainda verdadeiro) e deixou passar 12 substituições reais, concentradas em 4 pares escritos de formas muito diferentes.

**Curadoria de memórias editadas ou adicionadas à mão** (`optimize: true` em `POST /memory` e `PATCH /memory/{id}`, e o "Otimizar com IA" da interface). O modelo reescreve o texto livre em fatos curtos, independentes, na primeira pessoa e no estado atual; depois vêm a mesma proteção contra duplicatas e a mesma substituição usadas pela tool `remember`:

| Você escreve | É gravado |
|---|---|
| `meu cachorro thor e minha gata luna` | "Meu cachorro é o Thor." · "Minha gata é a Luna." |
| `Mudei de emprego, agora trabalho na Globant como dev sênior` | "Trabalho na Globant como dev sênior." |
| `reunião com o time amanhã 14h` | "Tenho uma reunião com o time em 28/09/2026 às 14h." |
| `i work at google as a data scientist and my wife is called emma` | "I work at Google as a data scientist." · "My wife is called Emma." |

Reescrever dados do usuário exige proteções, porque o modelo, sozinho, perdia detalhes ("como dev sênior"), traduzia notas em inglês e chegou a inventar fatos ("Tenho duas irmãs, Julia e Ana"). Por isso:
- O código confere se **toda palavra relevante da nota continua no resultado** e se **nada foi acrescentado** (no máximo um conectivo, como "se chama").
- Se a verificação falhar, há uma nova tentativa informando o problema; se falhar de novo, **o texto original é gravado**. Nenhum dado se perde por uma reescrita ruim, inclusive quando o Ollama cai no meio.
- Datas relativas ("amanhã", "ontem", "today") viram datas absolutas **no código**, antes de o modelo ver a nota, porque ele ignorou um calendário fornecido.

`python scripts/eval_curation.py` mede fidelidade, divisão e idioma (resultado atual: 30/30, mediana de 0,4 s por nota).

**Trocando de modelo: reindexar e calibrar.** Os três limiares acima dependem do modelo de embeddings, e a pergunta de substituição depende do modelo de chat. Trocar o `OLLAMA_EMBED_MODEL` também torna os vetores guardados incompatíveis: eles têm outra dimensão, e a busca passa a falhar. O Origin trata as duas coisas (**Configurações → Memória** na interface):

1. **Detecção:** `data/calibration.json` registra qual modelo indexou as memórias. Na inicialização, se o modelo atual for outro, ou se a dimensão dos vetores for diferente, o log avisa, a interface mostra "reindexar" no rodapé e no gerenciador de memória, e a busca responde **409** com a instrução, em vez de um 500 genérico.
2. **Reindexar** (`POST /memory/reindex`): recalcula o vetor de todas as memórias com o modelo atual, mantendo ids, metadados e arquivadas. Um backup em JSON é salvo em `data/backups/` antes, e a coleção antiga só é apagada depois que **todos** os vetores novos estiverem prontos. Se o Ollama cair no meio, nada se perde.
3. **Calibrar** (`POST /memory/calibration/run`): mede o modelo atual num conjunto embutido de pares rotulados (paráfrases, contradições, fatos compatíveis, sem relação, perguntas relevantes e irrelevantes, em pt/en) e sugere os limiares. Também testa a pergunta de substituição do modelo de chat atual: quantas contradições reconhece e se arquivaria algum fato ainda verdadeiro. Você revisa o relatório e aplica.

Os limiares aplicados valem só para o modelo de embeddings em que foram medidos. Ao trocar de modelo, o Origin volta aos padrões do `.env` até uma nova calibração.

Com os modelos atuais, a calibração automática reproduz os valores ajustados à mão (sugere 0,917 / 0,549 / 0,457 contra 0,92 / 0,55 / 0,45), e o eval do agente segue 100%. Com o `nomic-embed-text`, ela sugere 0,922 / 0,632 / 0,529 e avisa que as classes se sobrepõem, o que confirma que esse modelo é pior em português.

| Endpoint | Método | Descrição |
|---|---|---|
| `/memory/calibration` | GET | Modelo atual × modelo que indexou, dimensões, precisa reindexar?, limiares em uso, último relatório |
| `/memory/calibration/run` | POST | Mede e sugere `{apply?}` |
| `/memory/calibration/thresholds` | PUT / DELETE | Aplica limiares escolhidos / volta aos do `.env` |
| `/memory/reindex` | POST | Recalcula os vetores com o modelo atual (com backup) |

**Perguntas não salvam.** A tool `remember` declara `not_for_questions` nos metadados. Se a mensagem é só uma pergunta ("Onde eu moro?", "O que eu levo de presente pra ela?") e não tem um pedido explícito de salvar ("Você pode anotar que...?"), o agente descarta a chamada. Regras no prompt não bastavam: o modelo às vezes salvava "A capital da França é Paris." ou um plano tirado do histórico. Qualquer tool pode usar o mesmo mecanismo. Para esquecer algo de propósito ("esquece aquilo da alergia"), existe a tool `forget`, que apaga de verdade.

> O modelo de embeddings padrão é o `bge-m3` porque é multilíngue. Em testes com textos em português, o `nomic-embed-text` não separava fatos relevantes de irrelevantes. Se trocar de modelo, recalibre o `MEMORY_MIN_SCORE` e recrie a coleção, porque vetores de modelos diferentes não são compatíveis.

### Tools (agente)

O modelo decide sozinho quando chamar uma tool. O Origin executa a chamada, devolve o resultado ao modelo e repete o ciclo até ele responder, com no máximo `AGENT_MAX_TOOL_ITERATIONS` chamadas. Se uma tool falhar, o erro volta para o modelo em vez de derrubar a requisição.

| Tool | O que faz |
|------|-----------|
| `remember` | Salva um fato sobre você ("Lembre que...", "Mudei de..."), sem duplicar e arquivando o que ficou desatualizado |
| `forget` | Apaga uma memória por id ou descrição ("Esquece aquilo da...") |
| `get_current_datetime` | Data, hora e dia da semana locais, no idioma de `ORIGIN_LOCALE` |
| `days_until` | Quantos dias faltam até uma data conhecida (`MM-DD` = próxima ocorrência, ou `YYYY-MM-DD`) |
| `date_offset` | Qual é a data de um dia relativo a hoje ("ontem", "daqui a um ano", "há 3 meses"), com o dia da semana |

Modelos pequenos erram contas de calendário. Sem a `date_offset`, "daqui a um ano" virava uma data chutada e "ontem" vinha com o dia da semana errado: 3/18 acertos, contra 60/60 com a tool. A regra geral é que toda aritmética de datas acontece no código, nunca no modelo.

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

**Idioma.** As saídas das tools são em inglês e são a última coisa que o modelo lê antes de responder, o que fazia modelos pequenos responderem em inglês. Por isso, o agente detecta o idioma da mensagem (português, inglês ou espanhol, por palavras comuns) e anexa à saída da tool uma instrução **escrita nesse idioma**: responder por "você" e sem misturar palavras em inglês. Três tentativas anteriores falharam:
- citar a mensagem do usuário fazia o modelo repeti-la como resposta;
- uma instrução em inglês ("reply in Portuguese") ainda puxava palavras em inglês ("algo else");
- uma frase de exemplo na instrução fazia o modelo copiar os fatos do exemplo para respostas reais.

### Falhas transitórias do Ollama

Durante os testes, o processo que executa o modelo no Ollama (`llama-server`) caiu várias vezes e se recuperou sozinho segundos depois. O Origin tenta de novo automaticamente, em duas camadas (`OLLAMA_RETRY_ATTEMPTS=3`, espera de `OLLAMA_RETRY_BACKOFF=0.5` s que dobra a cada tentativa):

| Falha | Onde é tratada | Como |
|---|---|---|
| Resposta 500 no início da requisição (`health resp ... connectex`), 503, conexão recusada ou perdida | Transporte HTTP (`origin/retry.py`), usado por todos os clientes: chat, roteador, juiz e embeddings, síncronos e assíncronos | Reenvia a requisição. Erros reais, como modelo inexistente ou requisição inválida, não são repetidos |
| O processo morre **no meio** da resposta (`error reading llama-server response ... wsarecv`), depois de o Ollama já ter respondido 200 | Chamadas ao modelo | Chamadas internas (roteamento, juiz, normalização, verificação de afirmações) são repetidas inteiras. A resposta ao usuário só é repetida se **nenhum token** tiver sido enviado; se já saiu texto, o erro aparece, para nunca juntar duas respostas diferentes |

As camadas não se multiplicam: a segunda só trata falhas no meio do stream, porque as do início já foram repetidas pelo transporte. Num teste real, encerrando o `llama-server` durante uma geração, a chamada **falhou em 1,3 s sem retentativa** e **completou em 11,3 s com retentativa** (o Ollama recarrega o modelo nesse intervalo).

### Latência

Cada turno registra no log o tempo de cada fase (`Turn timings: context=… routing=… tools=… first_token=…`). Três correções saíram dessas medições:

| Causa | Correção | Efeito |
|---|---|---|
| `localhost` resolve primeiro para `::1`; no Windows, cada conexão nova espera ~2 s a tentativa IPv6 falhar | `OLLAMA_BASE_URL` padrão `127.0.0.1`, e `localhost` é normalizado automaticamente | primeira busca na memória: 2,1 s → 75 ms |
| Cada cliente (chat, roteador, juiz, embeddings) cria o cliente HTTP no primeiro uso (~250 ms) | Pré-aquecimento de todos os clientes na inicialização | primeiras requisições sem custo extra |
| A memória recuperada e a instrução de roteamento ficavam no prompt de sistema, antes das definições das tools, e invalidavam o cache do Ollama (~1.000 tokens reprocessados por chamada) | O prompt de sistema é estático; o que varia vai no fim da mensagem do usuário | roteamento ~200 ms mais rápido |

Primeiro token pelo `/chat/events`, em um servidor recém-iniciado: **~0,2 a 0,5 s** sem tools e **~0,8 a 0,9 s** com tools. Antes, era de 3,1 a 3,9 s nas primeiras requisições.

### Avaliação do agente

`scripts/eval_agent.py` roda casos reais contra os modelos locais, cada um com uma memória isolada. Os grupos são: `single` (uma intenção), `dates` (datas relativas a hoje), `compound` (várias intenções), `session` (não repetir ações de mensagens anteriores e não trocar de idioma), `question` (perguntas sobre fatos salvos são respondidas, não salvas de novo), `memory` (duplicatas, substituições e esquecimento) e `followup` (recuperar fatos pelo contexto). Todo caso também reprova respostas que só repetem a mensagem do usuário.

Resultado atual (qwen3:8b, 10 runs por caso): **420/420**. Use-o para comparar modelos, prompts e configurações antes de mudar um padrão:

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
