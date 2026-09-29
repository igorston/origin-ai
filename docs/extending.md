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

- `ctx` traz as configurações, a memória do usuário (com contas, a de cada um) e o modelo usado para verificações.
- As tools dos plugins entram depois das nativas. Uma tool com o nome de uma nativa é recusada, e não substitui a original.
- `TOOLS_DISABLED` desliga qualquer uma delas pelo nome.

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
