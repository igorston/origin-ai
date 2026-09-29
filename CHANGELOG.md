# Changelog

Formato baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/); o projeto segue [versionamento semântico](https://semver.org/lang/pt-BR/).

## [0.5.0] - 2026-09-29

### Segurança
- Regras em código contra *prompt injection* (`TurnGuard`, `AGENT_TOOL_GUARD=true`): `fetch_url` só abre endereços vindos do usuário ou de resultados do turno; depois de ler conteúdo de fora, `remember` e `forget` só rodam com pedido explícito do usuário; a verificação de afirmações ("anotei") passa pelas mesmas regras. A tool recusada recebe o motivo, e o modelo responde sem ela.

### Extensões
- Novos metadados de tool: `untrusted`, `side_effect` e `url_arg` ([docs/extending.md](docs/extending.md)). `web_search` e `fetch_url` são `untrusted`; `remember` e `forget` são `side_effect`.

## [0.4.2] - 2026-09-29

### Extensões
- A verificação `metadata["auto"]` pode receber também as tools que o roteamento escolheu (`fn(message, called)`), para ficar de fora quando outra tool já responde. A base de conhecimento usa isso: com o total de uma planilha já calculado, a busca de documentos ainda trazia linhas soltas, e o modelo as somava errado à mão.

## [0.4.1] - 2026-09-29

### Extensões
- Tools de plugins podem se chamar sozinhas quando o roteamento as deixa de fora (`metadata["auto"]`), como a pesquisa obrigatória de dados ao vivo. Perguntada sobre uma regra da empresa, a IA às vezes não consultava os documentos e inventava a política.

## [0.4.0] - 2026-09-29

### Corrigido
- Os dias da semana das tools de data seguem o idioma do app (o da marca, senão `ORIGIN_LOCALE`), vindos dos pacotes de idioma. Uma marca em pt-BR sem `ORIGIN_LOCALE` no `.env` recebia os nomes em inglês, e o modelo os traduzia errado ("Wednesday" virava "terça-feira"): 9 de 18 casos de datas.

### Extensões
- Tools de plugins podem pedir resposta completa (`metadata["reply"] = "detailed"`) e declarar as próprias fontes (`metadata["sources"]`), que entram na lista de "Fontes" quando a resposta não cita nenhuma.
- `ToolContext.workspace`: as tools sabem de qual usuário são, para dados e permissões por usuário.
- `render_index(..., template=...)` monta páginas de plugins com a marca, o idioma e os dados de inicialização do núcleo.
- Os links da marca aceitam caminhos do próprio site (`"/knowledge"`), abertos na mesma aba. `//outro-site` e esquemas como `javascript:` continuam recusados.

## [0.3.0] - 2026-09-29

### Corrigido
- O medidor de contexto não conta mais o resultado das ferramentas: depois de uma pesquisa na web, a página lida entrava na medição, o medidor mostrava 76% e as estimativas eram calibradas por esse número. Turnos com ferramentas passam a ser estimados.
- Acesso à web: a conexão vai para o endereço que foi conferido como público, e não para uma segunda consulta ao DNS. Um domínio que respondesse público na conferência e interno na conexão (*DNS rebinding*) passava pela proteção. O certificado continua conferido pelo nome do site.
- Números de lei com mais de um dígito ("lei 14.790", "PL 2338", "ley 27.430") não disparavam a pesquisa obrigatória.
- Importar `origin.memory.curator` antes do resto do pacote falhava por importação circular.
- O exemplo da tool `remember` usa marcadores ("Meu time favorito é o <time>."): com um nome real, o modelo às vezes salvava o exemplo no lugar do fato do usuário.

### Interface
- Tabelas nas respostas, com alinhamento por coluna e rolagem horizontal quando largas.
- Listas aninhadas por indentação.

### Idiomas
- Pacotes de idioma em `origin/i18n/agent/` (pt, en, es): instruções de resposta, reconhecimento do idioma e as frases que o agente reconhece (pesquisa obrigatória, pedidos de explicação, memória, histórias). Um idioma novo é um arquivo, sem mexer no código. As regras antigas foram transcritas sem mudança de comportamento, conferido em 1.463 frases dos testes e da avaliação.

### Qualidade
- Testes de interface num navegador real (Playwright, `pytest -m e2e`), com o modelo simulado, também no CI.
- Roteiro de lançamento no CONTRIBUTING.
- A avaliação não reprova mais respostas a saudações curtas que começam repetindo a saudação.

## [0.2.0] - 2026-09-29

### Extensões
- O Origin pode ser usado como dependência por outro projeto ([docs/extending.md](docs/extending.md)):
  - tools de pacotes instalados, pelo entry point `origin.tools`;
  - rotas e ajustes no app, pelo entry point `origin.app`;
  - um plugin que falha impede o app de subir, com o nome dele no erro.
- O app é montado por `origin.app.create_app()`, dentro do pacote. O `main.py` continua funcionando.
- Comandos `origin` e `python -m origin` para rodar o pacote instalado.

## [0.1.0] - 2026-09-28

Primeira versão pública.

### Assistente
- Engine sobre Ollama e LangChain, com streaming (texto e SSE) e um turno de roteamento que decide as ferramentas antes da resposta.
- Tools plugáveis: data e hora, dias até uma data, datas relativas (com a conta feita em código), lembrar e esquecer.
- Memória de longo prazo em Chroma (bge-m3):
  - deduplicação, substituição de fatos desatualizados e recuperação pelo contexto da conversa;
  - a substituição reconhece mudanças ditas de outro jeito ("mudei de time, agora torço pro Náutico" substitui "meu time favorito é o Sport") e mantém o que é acréscimo ("também gosto de...");
  - gerenciador na interface, com curadoria por IA ao editar;
  - calibração e reindexação quando o modelo muda.
- Histórias e poemas: um escritor dedicado planeja a história e escreve uma cena por vez, com título e capítulos, e remove repetições.
- Respostas formatadas: títulos, listas (inclusive logo abaixo de um título), separadores e links clicáveis, também nas fontes.
- Ao responder pelo que lembra, o assistente fala com o usuário ("você mora em..."), não como se fosse ele.
- Proteção contra troca de alfabeto: trechos em chinês, japonês ou coreano são traduzidos no meio do streaming.

### Internet
- Acesso à internet opcional para o agente: `web_search` (DuckDuckGo, SearXNG ou Brave) e `fetch_url` (lê páginas). Fica desligado até o usuário ligar o 🌐.
- Perguntas sobre dados que mudam (cotação, clima, notícias, placar) sempre geram uma pesquisa. Sem internet, o agente diz que não pode consultar em vez de inventar um valor.
- As fontes citadas são listadas com os links.
- Pedidos de explicação ("me explique a MP das Bets") leem a página do melhor resultado, pulando sites que bloqueiam robôs, e a resposta tem a extensão que o pedido pede, no idioma do usuário.
- Leis, decisões judiciais e medidas do governo também geram uma pesquisa, em vez de uma resposta de memória.
- Bloqueio de endereços locais e privados, conferido também a cada redirecionamento, e conteúdo da web marcado como não confiável.

### Contexto
- Medidor de contexto em relação à janela real, com a reserva para a resposta marcada.
- Janela automática, calculada pelo limite do modelo e pela VRAM (`OLLAMA_NUM_CTX=auto`).
- Otimização automática: as mensagens antigas viram um resumo em seções.
  - Os fatos do usuário e os textos escritos são preservados literalmente.
  - Os assuntos só são condensados quando isso libera espaço.
- A conversa só é encerrada quando o conteúdo preservado não cabe mais na janela. Depois disso, ela continua numa conversa nova, que leva o resumo.

### White label
- Marca configurável (`brand/brand.json`): nomes, logo, favicon, cores, persona, textos iniciais e links.
- Interface e mensagens do servidor em pt-BR e en. Um idioma novo é um arquivo novo em `origin/i18n/`.

### Distribuição
- Configuração inicial na interface, que baixa os modelos que faltam.
- A instalação padrão roda em modo `prod`; `ORIGIN_ENV=dev` liga o recarregamento automático para quem desenvolve.
- Instaladores para Windows e Linux/macOS, imagem Docker e Docker Compose (CPU e GPU NVIDIA).
- Contas opcionais (`ORIGIN_AUTH=password`):
  - login, gerenciamento de usuários pela linha de comando;
  - conversas e memórias isoladas por usuário;
  - cookies assinados, limite de tentativas e bloqueio de requisições de outras origens.
- CI com os testes em Python 3.11 a 3.14 e o build da imagem.

[0.5.0]: https://github.com/igorston/origin-ai/releases/tag/v0.5.0
[0.4.2]: https://github.com/igorston/origin-ai/releases/tag/v0.4.2
[0.4.1]: https://github.com/igorston/origin-ai/releases/tag/v0.4.1
[0.4.0]: https://github.com/igorston/origin-ai/releases/tag/v0.4.0
[0.3.0]: https://github.com/igorston/origin-ai/releases/tag/v0.3.0
[0.2.0]: https://github.com/igorston/origin-ai/releases/tag/v0.2.0
[0.1.0]: https://github.com/igorston/origin-ai/releases/tag/v0.1.0
