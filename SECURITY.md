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
- **Ferramentas:** as tools do agente rodam com as permissões do servidor. Revise uma tool nova antes de habilitá-la, principalmente se ela acessar arquivos, a rede ou o shell.
- **Modelos:** o modelo pode ser induzido, por conteúdo malicioso na conversa, a chamar ferramentas. As tools incluídas só leem datas e gravam na memória do próprio usuário.
