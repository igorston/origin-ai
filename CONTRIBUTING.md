# Contribuindo

Obrigado pelo interesse! Issues e pull requests são bem-vindos.

## Ambiente

```bash
git clone https://github.com/igorston/origin-ai.git && cd origin-ai
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env                              # e mude para ORIGIN_ENV=dev: reinicia a cada alteração
ollama pull qwen3:8b && ollama pull bge-m3        # para rodar de verdade e os testes de integração
```

## Antes de abrir um PR

```bash
ruff check . && ruff format --check .
pytest                     # os testes de integração pulam sozinhos sem Ollama
```

Os testes de interface (`tests/e2e`) abrem um navegador real contra o app, com o modelo e o Ollama simulados:

```bash
pip install -e ".[dev,e2e]" && playwright install chromium
pytest -m e2e              # ou ORIGIN_E2E_CHANNEL=msedge para usar o Edge instalado
```

Sem o Playwright ou sem um navegador eles pulam; no CI, falham.

O CI roda os testes em Python 3.11 a 3.14. Atenção a recursos exclusivos do 3.14: até o 3.13, as anotações de tipo são avaliadas na hora.

Para mudanças de comportamento do agente, rode também as avaliações com os modelos reais e cite os números no PR:

```bash
python scripts/eval_agent.py --runs 3      # ferramentas, datas, memória, perguntas compostas
python scripts/eval_context.py             # fatos que sobrevivem à otimização de contexto
```

## Convenções

- [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.
- **Comportamento de modelo é medido, não suposto.** Se um prompt resolve um problema, mostre antes e depois. Quando o modelo não é confiável, prefira uma proteção em código testável: foi assim com datas, perguntas compostas e troca de alfabeto.
- **Textos de interface ficam nos catálogos** (`origin/i18n/*.json`), nunca no código. Toda chave nova precisa existir em todos os idiomas; `tests/unit/test_i18n.py` confere.
- **Frases que o agente reconhece ficam nos pacotes de idioma** (`origin/i18n/agent/*.json`), nunca em regex no código: assim um idioma novo não precisa mexer em Python.
- **Prompts ficam em `origin/prompts/templates/`,** em inglês (o modelo segue melhor), com as respostas no idioma do usuário.

## Onde mexer

| Quero… | Veja |
|---|---|
| Criar uma tool | `origin/integrations/tools/` (cada arquivo é descoberto sozinho) |
| Adicionar um idioma | [docs/white-label.md](docs/white-label.md#2-idiomas) |
| Mudar a otimização de contexto | `origin/core/context.py` e `scripts/eval_context.py` |
| Mudar a interface | `origin/web/static/` (HTML, CSS e JS puros, sem build) |
| Testar a interface | `tests/e2e/` (Playwright; casos de Markdown em `markdown_cases.json`) |
| Estender por outro pacote | [docs/extending.md](docs/extending.md) (entry points `origin.tools` e `origin.app`) |

## Lançando uma versão

O CI não tem GPU, então a avaliação com o modelo real roda na máquina de quem lança:

1. `python scripts/eval_agent.py --web --runs 3` e `python scripts/eval_context.py`, sem regressão em relação à versão anterior.
2. `ORIGIN_E2E_CHANNEL=msedge pytest -m e2e` (ou com o Chromium do Playwright).
3. A versão em `pyproject.toml` e em `origin/__init__.py`, e a seção dela no `CHANGELOG.md`, com a data.
4. Commit, CI verde, e a tag: `git tag -a vX.Y.Z -m "..." && git push origin vX.Y.Z`.
5. No GitHub, um Release a partir da tag, com a seção do CHANGELOG como notas.

## Segurança

Vulnerabilidades: veja [SECURITY.md](SECURITY.md). Não abra issue pública.
