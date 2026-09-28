# Changelog

Formato baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/); o projeto segue [versionamento semântico](https://semver.org/lang/pt-BR/).

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

[0.1.0]: https://github.com/igorston/origin-ai/releases/tag/v0.1.0
