# White label: seu produto sobre o Origin

O Origin pode ser distribuído com outro nome, outra identidade visual e outra personalidade, sem mudar o código. Tudo fica num arquivo de marca e nas imagens ao lado dele.

## 1. Crie o arquivo de marca

```bash
cp brand/brand.example.json brand/brand.json
```

O caminho vem de `ORIGIN_BRAND_PATH` (padrão `./brand/brand.json`). As imagens citadas no arquivo ficam na mesma pasta. Reinicie o servidor para aplicar as mudanças.

Se o arquivo não existir, o Origin usa a marca padrão. Se o arquivo estiver inválido, o log avisa e o Origin volta à marca padrão; um erro de digitação nunca impede o servidor de subir.

| Campo | Para quê | Exemplo |
|---|---|---|
| `product_name` | Título da página, logo, título da API em `/docs`, logs | `"Aurora"` |
| `assistant_name` | Como o assistente se apresenta ("You are Aurora…") e o placeholder da mensagem | `"Aurora"` |
| `description` | Descrição da API em `/docs` | `"Assistente do escritório Silva"` |
| `logo` | Um emoji ou texto curto, **ou** um arquivo de imagem na pasta da marca (`.svg`, `.png`, `.webp`…) | `"logo.svg"` |
| `favicon` | Imagem do ícone da aba. Se ficar vazio, usa o `logo` | `"favicon.png"` |
| `locale` | Idioma da interface e das mensagens do servidor. Se ficar vazio, usa `ORIGIN_LOCALE` | `"pt-BR"` |
| `persona` | Papel, tom e público do assistente. Entra **depois** das regras que fazem o agente funcionar, e não no lugar delas | `"Seja formal…"` |
| `theme.light` / `theme.dark` | Cores (variáveis CSS) para o tema claro e o escuro | `{"accent": "#0f766e"}` |
| `welcome.title`, `welcome.text`, `welcome.suggestions` | Tela da conversa vazia. Se ficarem vazios, usam os textos do idioma | |
| `links` | Links no rodapé da barra lateral (só `http(s)://`) | `{"Suporte": "https://…"}` |

### Cores

Tokens aceitos em `theme.light` e `theme.dark`: `bg`, `surface`, `surface-2`, `border`, `text`, `muted`, `accent`, `accent-contrast`, `accent-soft`, `danger`, `ok`, `warn`, `user-bubble`, `user-text`, `code-bg`, `radius`.

Os valores só podem ser cores, medidas ou funções de cor, porque são validados antes de entrar na página. Se você definir só o `light`, o mesmo valor também vale no tema escuro. Defina o `dark` para ter uma variação própria.

### Persona

A persona muda o comportamento do assistente, e não só a aparência. Algumas dicas:

- Descreva quem o assistente é, para quem fala e o que deve ou não fazer ("não substitui um parecer jurídico").
- Não repita regras que o Origin já garante (idioma, uso de ferramentas, datas).
- Teste com perguntas reais. O modelo padrão é pequeno (8B) e segue melhor instruções curtas e concretas.

A persona **não** é enviada ao navegador: nem `GET /api/brand` nem a página a expõem, e a rota `/brand/…` serve apenas imagens.

## 2. Idiomas

A interface vem em **pt-BR** e **en**. Cada idioma é um arquivo em `origin/i18n/` com duas seções: `web` (a interface) e `server` (mensagens da API).

Para adicionar um idioma:

1. Copie `origin/i18n/en.json` para `origin/i18n/<código>.json` (por exemplo `es.json`, `fr.json`) e traduza os valores. Não altere as chaves nem os marcadores `{nome}`.
2. Troque `"_name"` pelo nome do idioma na própria língua ("Español").
3. Rode `pytest tests/unit/test_i18n.py`. O teste confere que o arquivo tem exatamente as mesmas chaves e marcadores dos outros.

O idioma aparece sozinho no seletor em Configurações. Cada navegador guarda a sua escolha. As respostas do assistente seguem o idioma em que o usuário escreve, qualquer que seja o da interface.

## 3. Modelos

Os modelos são configuráveis (`OLLAMA_MODEL`, `OLLAMA_EMBED_MODEL`). Ao trocar de modelo:

- **Janela de contexto:** é recalculada no início (`OLLAMA_NUM_CTX=auto`).
- **Memória:** use **Configurações → Memória → Reindexar / Calibrar** para ajustar os limiares ao novo modelo de embeddings.
- **Licença:** confira a do modelo antes de distribuir. `qwen3` (Apache 2.0) e `bge-m3` (MIT) permitem uso comercial. Outros, como os Llama, têm termos próprios.

## 4. Obrigações da licença

O Origin é distribuído sob a [Apache 2.0](../LICENSE):

- **Pode:** usar, modificar, rebatizar e vender, inclusive em produto fechado.
- **Deve:** incluir uma cópia da licença e manter o arquivo [NOTICE](../NOTICE) (com os créditos de autoria) nas distribuições. Arquivos que você modificar devem indicar a mudança.
- **Não recebe:** direito sobre o nome "Origin" (seção 6 da licença). Use a sua própria marca, que é justamente o objetivo do white label.

## Checklist

- [ ] `brand/brand.json` com nome, logo e cores
- [ ] Persona testada com perguntas reais do seu público
- [ ] Idioma padrão (`locale` ou `ORIGIN_LOCALE`)
- [ ] Modelos escolhidos, e licenças conferidas
- [ ] LICENSE e NOTICE incluídos na distribuição
- [ ] Para servir a outras pessoas: [implantação](deployment.md) com login e HTTPS
