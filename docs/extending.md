# Estendendo o Origin com um pacote próprio

Um projeto pode usar o Origin como dependência e acrescentar tools, rotas e a própria marca, sem copiar nem alterar o código do Origin. As correções do núcleo chegam com uma atualização da versão.

## 1. Dependência

No `pyproject.toml` do seu projeto, fixe uma versão:

```toml
[project]
dependencies = ["origin-ai @ git+https://github.com/igorston/origin-ai.git@v0.2.0"]
```

## 2. Tools

Um módulo com `get_tools(ctx)`, igual aos de `origin/integrations/tools/`, declarado no grupo `origin.tools`:

```toml
[project.entry-points."origin.tools"]
crm = "acme_origin.tools.crm"          # um módulo com get_tools(ctx)
# ou uma função: crm = "acme_origin.tools.crm:tools_for"
```

```python
# acme_origin/tools/crm.py
from langchain_core.tools import BaseTool, tool

from origin.integrations import ToolContext


def get_tools(ctx: ToolContext) -> list[BaseTool]:
    @tool
    def crm_lookup(customer: str) -> str:
        """Look a customer up in the CRM."""
        ...

    return [crm_lookup]
```

- `ctx` traz as configurações, a memória do usuário (com contas, a de cada um), o modelo usado para verificações e `ctx.workspace`: de quem são essas tools (`""` sem contas e para o primeiro usuário, `"u<id>"` para os outros). Use-o para dados por usuário e permissões.
- As tools dos plugins entram depois das nativas. Uma tool com o nome de uma nativa é recusada, e não substitui a original.
- `TOOLS_DISABLED` desliga qualquer uma delas pelo nome.

Metadados da tool (`tool.metadata = {...}`) ajustam o que o núcleo faz com o resultado:

| Chave | Efeito |
| --- | --- |
| `"reply": "detailed"` | A resposta segue o tamanho do pedido, como após a web. Sem isso, após uma tool o modelo é instruído a responder em uma ou duas frases, o que serve para confirmar uma ação, não para explicar um documento. |
| `"sources": fn` | `fn(output) -> ["Manual.pdf, p. 3", ...]`, as fontes do resultado, da melhor para a pior. Se a resposta não citar nenhuma, as três primeiras entram na lista de "Fontes" sob ela. |
| `"auto": fn` | `await fn(message) -> dict \| None`: os argumentos com que a tool se chama sozinha quando o roteamento não a chamou (ou `None`). Serve de rede de segurança, como a pesquisa obrigatória de dados ao vivo: a base de conhecimento se chama quando a mensagem tem um documento muito parecido. Com dois parâmetros, `fn(message, called)` recebe também os nomes das tools que o roteamento escolheu. Uma falha em `fn` é registrada no log e ignorada. |
| `"network": True` | A tool usa a internet: só é oferecida quando o usuário liga o 🌐. |
| `"not_for_questions": True` | Não é executada para perguntas puras ("Onde eu moro?"), a menos que `"explicit_intent"` (uma regex) apareça na mensagem. |
| `"untrusted": True` | O resultado é conteúdo de fora (página, documento, célula de planilha) que qualquer um pode ter escrito. Depois que uma tool assim roda, o turno fica "contaminado". Marque toda tool que traz texto de terceiros. |
| `"side_effect": True` | A tool altera dados do usuário (grava, apaga, envia). Num turno contaminado, só roda se a mensagem do próprio usuário pedir a ação (`"explicit_intent"` casa); senão é recusada e o modelo recebe o motivo. |
| `"url_arg": "url"` | Esse argumento é um endereço a abrir. Só é aceito se veio do usuário (mensagens da conversa) ou de um resultado de tool neste turno (busca, link de página lida); um endereço composto pelo modelo é recusado. |

As três últimas são regras em código contra *prompt injection* (`origin.core.agent.TurnGuard`, ligadas por `AGENT_TOOL_GUARD`): o modelo pode ser convencido a ignorar uma instrução do prompt, mas não essas.

## 3. Rotas e ajustes no app

Uma função `register(app, settings)` no grupo `origin.app`. Ela recebe o app já com as rotas nativas e roda antes da interface web ser montada:

```toml
[project.entry-points."origin.app"]
acme = "acme_origin.app:register"
```

```python
# acme_origin/app.py
from fastapi import APIRouter, FastAPI

from origin.config import Settings

router = APIRouter(prefix="/api/acme", tags=["acme"])


@router.get("/status")
def status() -> dict[str, str]:
    return {"plan": "enterprise"}


def register(app: FastAPI, settings: Settings) -> None:
    app.include_router(router)
```

Com as contas ligadas (`ORIGIN_AUTH=password`), as rotas do plugin exigem login como todas as outras. Só a tela de login, a marca e o `/health` são públicos.

### Uma página própria

`origin.web.render_index(settings, user, template=Path(...))` monta uma página HTML do plugin como as do núcleo: com a marca (título, favicon, cores), o idioma e os dados de inicialização. O modelo usa os marcadores `{{lang}}`, `{{title}}`, `{{favicon}}`, `{{theme}}` e `{{boot}}`, e pode importar `/static/styles.css` e `/static/i18n.js`. Para um link na barra lateral do chat, use os `links` da marca com um caminho do próprio site (`"/knowledge"`), que abre na mesma aba.

## 4. Executar

Na pasta do projeto, com o `.env` e o `brand/` dele:

```bash
origin              # ou: python -m origin
```

O comando `origin` monta o app com `origin.app:create_app`, que carrega os plugins instalados. No Windows, o Smart App Control pode bloquear o `origin.exe` gerado pelo pip. Nesse caso, use `python -m origin`, que faz o mesmo.

Para montar o app no próprio código (testes, outro servidor ASGI):

```python
from origin.app import create_app

app = create_app()
```

## Falhas

Um plugin que não carrega, ou cujo `register` falha, impede o app de subir, e o erro traz o nome do plugin. Ele foi instalado de propósito, e rodar sem ele esconderia o problema.
