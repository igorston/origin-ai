# Contribuindo

Obrigado pelo interesse! Issues e pull requests são bem-vindos.

## Ambiente

```bash
git clone https://github.com/igorston/origin-ai.git && cd origin-ai
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env
ollama pull qwen3:8b && ollama pull bge-m3        # para rodar de verdade e os testes de integração
```

## Antes de abrir um PR

```bash
ruff check . && ruff format --check .
pytest                     # os testes de integração pulam sozinhos sem Ollama
```

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
- **Prompts ficam em `origin/prompts/templates/`,** em inglês (o modelo segue melhor), com as respostas no idioma do usuário.

## Onde mexer

| Quero… | Veja |
|---|---|
| Criar uma tool | `origin/integrations/tools/` (cada arquivo é descoberto sozinho) |
| Adicionar um idioma | [docs/white-label.md](docs/white-label.md#2-idiomas) |
| Mudar a otimização de contexto | `origin/core/context.py` e `scripts/eval_context.py` |
| Mudar a interface | `origin/web/static/` (HTML, CSS e JS puros, sem build) |

## Segurança

Vulnerabilidades: veja [SECURITY.md](SECURITY.md). Não abra issue pública.
