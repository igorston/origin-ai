# Segurança

## Como reportar

Encontrou uma vulnerabilidade? **Não abra uma issue pública.** Use o [aviso de segurança privado do GitHub](https://github.com/igorston/origin-ai/security/advisories/new). Responderemos o quanto antes e daremos crédito na correção, se você quiser.

## Modelo de ameaça

- **Padrão local:** o Origin escuta só em `127.0.0.1` e não tem login (`ORIGIN_AUTH=off`). Qualquer programa da própria máquina pode usar a API. Isso é adequado para uso pessoal num computador de confiança.
- **Com a rede:** ao expor o servidor, ligue as contas (`ORIGIN_AUTH=password`) e coloque HTTPS na frente ([docs/deployment.md](docs/deployment.md)). O Origin não faz TLS.
- **Com contas ligadas:**
  - senhas guardadas com scrypt;
  - sessões em cookies assinados com HMAC-SHA256, com HttpOnly e SameSite=Lax;
  - requisições que alteram dados vindas de outra origem são recusadas;
  - tentativas de login limitadas por endereço;
  - conversas e memórias isoladas por usuário.
- **Internet:** as tools de web só existem com `WEB_ACCESS=true` e só são oferecidas ao modelo quando o usuário liga o acesso. Elas só leem endereços públicos: localhost, redes privadas, link-local (metadados de nuvem) e redirecionamentos para eles são recusados, o que evita SSRF. `WEB_ALLOW_PRIVATE=true` desliga essa proteção e deve ser evitado em servidores compartilhados. Uma limitação conhecida: a checagem é feita antes da conexão, então um DNS que muda entre a checagem e a conexão (*DNS rebinding*) não é coberto; para isolamento total, rode o Origin numa rede sem acesso interno. O conteúdo das páginas vai para o modelo marcado como não confiável; as regras abaixo limitam o que ele pode causar.
- **Ferramentas:** as tools do agente rodam com as permissões do servidor. Revise uma tool nova antes de habilitá-la, principalmente se ela acessar arquivos, a rede ou o shell.
- **Modelos:** o modelo pode ser induzido, por conteúdo malicioso na conversa, a chamar ferramentas. As tools incluídas leem datas, gravam na memória do próprio usuário e, com a internet ligada, pesquisam e leem páginas públicas; nenhuma apaga arquivos nem executa comandos.
- ***Prompt injection*:** uma página ou documento pode trazer instruções escritas para o assistente. Instruções no prompt não bastam (o modelo pode ser convencido a ignorá-las), então três regras valem em código, a cada turno (`AGENT_TOOL_GUARD=true`, o padrão):
  - **Procedência de endereços:** `fetch_url` só abre endereços que vieram do usuário ou de um resultado deste turno (busca, link de uma página lida). Um endereço montado pelo modelo, que poderia levar dados do usuário ao servidor de outra pessoa, é recusado.
  - **Contaminação:** depois que conteúdo de fora é lido, as tools que alteram dados (`remember`, `forget`) só rodam se a mensagem do próprio usuário pedir a ação ("guarde", "esquece").
  - **Verificação de afirmações:** a checagem que executa uma ação que a resposta diz ter feito ("anotei") passa pelas mesmas regras, então uma página não consegue fazer o assistente "cumprir" o que ela mandou dizer.
  - O que continua possível: a página ainda pode influenciar o **texto** da resposta (informação falsa, um link sugerido). Trate o que vem da web e de documentos como qualquer fonte externa. Tools novas que trazem texto de terceiros ou alteram dados devem declarar os metadados `untrusted` / `side_effect` / `url_arg` ([docs/extending.md](docs/extending.md)).
