# Implantação

O Origin roda de três jeitos. Escolha pelo público:

| Jeito | Para quem | Login |
|---|---|---|
| **Local** (Python) | Uma pessoa, no próprio computador | Desligado (padrão) |
| **Docker Compose** | Uma máquina dedicada, ou para empacotar para outras pessoas | Opcional |
| **Servidor compartilhado** | Várias pessoas, pela rede | **Obrigatório**, com HTTPS |

Nos três casos, os modelos rodam no [Ollama](https://ollama.com), na mesma máquina ou numa máquina com GPU acessível pela rede (`OLLAMA_BASE_URL`).

## Local

```bash
sh scripts/install.sh                                          # Linux / macOS
powershell -ExecutionPolicy Bypass -File scripts\install.ps1   # Windows
```

Os instaladores criam o ambiente Python, instalam as dependências, criam o `.env` e baixam os modelos com o Ollama (`--skip-models` / `-SkipModels` pula esta etapa). Depois é só executar `python main.py` e abrir http://127.0.0.1:8000.

Se faltar algum modelo, a interface abre sozinha a **Configuração inicial**, que baixa os modelos com uma barra de progresso.

O servidor escuta só em `127.0.0.1`: ninguém mais na rede o acessa.

## Docker Compose

```bash
docker compose up -d                                                   # CPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d   # GPU NVIDIA
```

- Sobe o Origin e o Ollama. A primeira abertura de http://127.0.0.1:8000 mostra a configuração inicial, que baixa os modelos para o volume do Ollama.
- **Dados:** conversas, memórias e calibração ficam no volume `origin-data`, e os modelos no volume `ollama`. Para backup, copie esses volumes.
- **Marca:** vem da pasta `./brand` (montada somente leitura).
- **Configurações:** vão em `environment:` no `docker-compose.yml`. Não use o `.env` local, porque os caminhos dele (`./data/...`) ficariam fora do volume.
- **GPU:** requer o [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) no Linux, ou o Docker Desktop com WSL 2 no Windows. Sem GPU visível para o Origin, a janela automática cai para 4096 tokens; nesse caso, defina `OLLAMA_NUM_CTX` à mão.
- **Porta:** é publicada só em `127.0.0.1`. Para abrir para a rede, veja a próxima seção.

## Servidor compartilhado

### 1. Ligue as contas

```bash
ORIGIN_AUTH=password
ORIGIN_ADMIN_USER=admin
ORIGIN_ADMIN_PASSWORD=troque-esta-senha   # cria o primeiro admin, se ainda não houver usuários
```

Com contas ligadas:

- **Isolamento:** cada usuário tem **as próprias conversas e a própria memória**.
- **Dados existentes:** o primeiro usuário (id 1) herda os dados que já existiam, então ligar o login não perde nada.
- **Permissões:** só administradores podem baixar modelos pela configuração inicial.
- **Sessão:** fica num cookie assinado (HttpOnly, SameSite=Lax), válido por `ORIGIN_SESSION_DAYS` (30). Trocar a senha encerra todas as sessões daquele usuário.
- **Força bruta:** as tentativas de login são limitadas a 10 falhas em 10 minutos por endereço.
- **Outras origens:** requisições de outros sites, com cookies de carona, são recusadas.

Gerencie os usuários pela linha de comando, no mesmo ambiente do servidor (com o mesmo `SQLITE_PATH`):

```bash
python -m origin.auth add-user maria --admin     # pede a senha
python -m origin.auth add-user joao
python -m origin.auth passwd joao
python -m origin.auth list
python -m origin.auth remove joao                # os dados dele continuam no disco
echo "senha-longa" | python -m origin.auth add-user ana --password-stdin
```

No Docker: `docker compose exec origin python -m origin.auth add-user maria --admin`.

A chave que assina os cookies é gerada e guardada em `secret.key`, ao lado do banco. Para fixá-la (por exemplo, com várias réplicas), defina `ORIGIN_SECRET_KEY`.

Sobre a internet: o acesso do agente à web (`web_search`, `fetch_url`) fica disponível, mas cada usuário precisa ligá-lo (🌐). Para proibi-lo no servidor, defina `WEB_ACCESS=false`. Mantenha `WEB_ALLOW_PRIVATE=false`, que é o padrão; com ele, os usuários não conseguem fazer o agente ler endereços da rede interna do servidor. Para buscas sem depender de terceiros, use uma instância própria do [SearXNG](https://docs.searxng.org/) (`WEB_SEARCH_PROVIDER=searxng`).

### 2. Coloque HTTPS na frente

O Origin não faz TLS. Use um proxy reverso. Com o [Caddy](https://caddyserver.com), que emite o certificado sozinho:

```
assistente.seudominio.com {
    reverse_proxy 127.0.0.1:8000
}
```

Defina também `ORIGIN_COOKIE_SECURE=true`, para que o cookie de sessão só trafegue por HTTPS. O streaming das respostas usa SSE. O Caddy o repassa sem configuração extra; no nginx, desligue o buffering (`proxy_buffering off;`).

### 3. Dimensione

- **GPU:** um modelo de 8B atende várias pessoas, mas **uma resposta por vez** por GPU. As outras esperam na fila do Ollama. Para mais simultaneidade, veja `OLLAMA_NUM_PARALLEL` no Ollama (cada slot extra consome a janela de contexto inteira em VRAM) ou use GPUs adicionais.
- **Janela:** o cálculo automático considera a VRAM livre no início. Outros programas que ocupem a GPU depois reduzem a folga.

## Atualizações

```bash
git pull && pip install -e .        # local
docker compose up -d --build        # Docker
```

As migrações do banco rodam sozinhas no início, e as mudanças estão no [CHANGELOG](../CHANGELOG.md). Faça backup de `data/` (ou do volume `origin-data`) antes de atualizar.
