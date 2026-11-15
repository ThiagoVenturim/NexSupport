# Support Intelligence Engine

Inteligencia de suporte sobre chamados e tickets de help desk. Faz a ingestao de tickets
de qualquer plataforma para um modelo unico, constroi uma base de conhecimento curada
com RAG hibrido para copiloto de atendimento, e gera analises operacionais: sentimento,
tendencias com alertas, radar de risco, qualidade, causa raiz, desvio e ROI.

Modelos (abertos ou pagos), rastreamento com Langfuse e primitivas de recuperacao vem de
`nexcoder-core` (pacote `packages/core` do monorepo nexcoder).

Status: em desenvolvimento ativo. Todos os dados de exemplo neste repositorio sao sinteticos.

## Conectores

Cada origem de tickets normaliza os registros para `support_engine.Ticket`:

- `ZendeskConnector`: API v2 do Zendesk Support (`/api/v2/tickets.json` e comentarios).
- `FreshdeskConnector`: API v2 do Freshdesk (`/api/v2/tickets` e conversas).
- `JiraSMConnector`: REST API v3 do Jira Service Management / Jira Cloud (`/rest/api/3/search` com extracao de Atlassian Document Format).
- `IntercomConnector`: Conversations REST API do Intercom (`/conversations` e `conversation_parts`).
- `CSVConnector` e `JSONConnector` (`load_tickets_csv`, `load_tickets_json`, `load_tickets_file`): arquivos locais CSV, JSON e JSONL protegidos por `nexcoder_core.resolve_inside`.

```python
from support_engine import ZendeskConnector, load_tickets_csv

connector = ZendeskConnector(base_url="https://acme.zendesk.com")
ticket = connector.to_ticket(
    {
        "id": 1042,
        "subject": "Erro ao exportar fatura mensal",
        "description": "Clicar em Exportar PDF retorna HTTP 500.",
        "status": "solved",
        "priority": "high",
        "created_at": "2026-02-10T14:00:00Z",
        "satisfaction_rating": {"score": "good"},
    }
)

csv_tickets = load_tickets_csv("tests/fixtures/tickets_sample.csv")
```

## Armazenamento e Banco Plugavel

O Support Intelligence Engine possui uma camada de armazenamento independente de banco
construida sobre SQLAlchemy Core com queries parametrizadas:

- **Padrao:** SQLite (`sqlite:///support_engine.db`) para zero configuracao.
- **Banco proprio:** configure a variavel `DB_URL` no `.env` ou no ambiente apontando para PostgreSQL (`.[postgres]`), MySQL (`.[mysql]`) ou SQL Server ODBC (`.[mssql]`).
- **Mascaramento de PII:** redacao automatica de e-mails, CPF, CNPJ, telefones, cartoes de credito e chaves de API.

```python
from support_engine import TicketStore, mask_ticket

with TicketStore() as store:
    store.save_ticket(ticket, mask_pii=True)
    saved = store.get_ticket("1042")
```

Consulte [docs/DATABASE.md](docs/DATABASE.md) para detalhes completos de configuracao.

## Linha de Comando (CLI)

```bash
# Importar tickets para o banco com mascaramento de PII
support-engine import --input tests/fixtures/tickets_sample.csv --mask-pii

# Listar tickets armazenados
support-engine list --status open

# Exibir detalhes e comentarios de um ticket
support-engine show --id T-101

# Exibir resumo estatistico
support-engine stats
```

## Conteineres e Scripts Multiplataforma

- **Docker Compose:** `docker compose up` roda a aplicacao com armazenamento local; `docker compose --profile postgres up` sobe um conteiner PostgreSQL junto com o app.
- **Kubernetes:** manifestos de exemplo em `k8s/` (`deployment.yaml`, `service.yaml`, `configmap.yaml`, `secret.yaml`).
- **Scripts:** `.\scripts\run.ps1` (PowerShell no Windows) e `./scripts/run.sh` (Bash no Linux / macOS).

## Estrutura do Repositorio

- `src/support_engine/`: o pacote Python.
- `tests/`: testes e massas de dados sinteticas (execucao offline).
- `k8s/`: manifestos Kubernetes.
- `scripts/`: scripts multiplataforma de execucao e verificacao.
- `docs/`, `examples/`: guias e exemplos executaveis.

## Desenvolvimento

```bash
python -m venv .venv
.venv/bin/pip install -e path/to/nexcoder/packages/core
.venv/bin/pip install -e ".[dev]"
git config core.hooksPath .githooks
.venv/bin/python -m pytest
.venv/bin/ruff check .
python scripts/check_no_emoji.py
```

## Origem

Evoluido a partir da arquitetura de suporte inteligente inicialmente construida
dentro da plataforma ApoIA e do lochome, reescrita como um pacote de codigo aberto
limpo, independente de plataforma e com fixtures sinteticas.

## Licenca

MIT. Veja `LICENSE`.
