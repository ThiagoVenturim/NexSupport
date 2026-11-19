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
- **Mascaramento de PII:** ligado por padrao antes de gravar (`mask_pii=False` para desligar); redige e-mails, CPF, CNPJ, telefones, cartoes de credito e chaves de API.

```python
from support_engine import TicketStore, mask_ticket

with TicketStore() as store:
    store.save_ticket(ticket)  # mascarado por padrao
    saved = store.get_ticket("1042")
```

Consulte [docs/DATABASE.md](docs/DATABASE.md) para detalhes completos de configuracao.

## Qualidade da Ingestao

`support_engine.ingestion` limpa as conversas (HTML, respostas citadas,
assinaturas, rodapes, avisos legais), mascara PII incluindo nomes de pessoas,
remove interacoes repetidas e divide cada ticket em chunks por interacao mais um
resumo do caso, com remocao de quase duplicados por MinHash em todo o corpus.
Todas as etapas sao configuraveis e rodam offline.

```python
from support_engine.ingestion import ChunkingConfig, IngestionConfig, IngestionPipeline

config = IngestionConfig(chunking=ChunkingConfig(max_chars=800, overlap_chars=100))
result = IngestionPipeline(config).process(tickets)
print(result.stats.pii_masked, len(result.chunks))
```

Veja [docs/INGESTION.md](docs/INGESTION.md) para todas as opcoes e os padroes de privacidade.

## Curadoria de Respostas

`support_engine.curation` da a cada resposta de ticket um score de resolucao
(status resolvido, reaberturas, CSAT, um juiz LLM opcional e avaliacoes humanas),
grava esse score nos metadados dos chunks para que a busca prefira respostas
comprovadas, transforma correcoes humanas em novas entradas do indice que
substituem a resposta antiga e sugere um artigo da base de conhecimento quando a
mesma pergunta sem resposta se repete. Metricas de qualidade resumem cada
execucao. Pesos e limiares sao configuraveis.

```python
from support_engine.curation import CurationPipeline, Feedback

feedback = [Feedback.vote("T-101", helpful=False, correction="Reset the token, then sign in again.")]
result = CurationPipeline().curate(ingested.tickets, ingested.chunks, feedback)
print(result.metrics.mean_score, len(result.suggestions))
```

Veja [docs/CURATION.md](docs/CURATION.md) para a formula do score e todas as opcoes.

## Agente de Atendimento

`support_engine.agent` responde perguntas de clientes a partir da base curada,
em um grafo LangGraph: guardas de entrada (mascara PII, bloqueia prompt injection,
recusa perguntas fora de escopo ou longas demais), cache semantico, resposta com
citacoes das fontes recuperadas, verificacao das citacoes, juiz com nova tentativa
explicada e guardas de saida. O resultado fica em um de tres niveis: 1 responde
ao cliente, 2 vira rascunho para revisao de um atendente, 3 escala para uma pessoa.

```python
from support_engine.agent import SupportAgent

agent = SupportAgent.from_chunks(curated.chunks, curation=curation)
result = agent.answer("A exportacao de faturas falha com erro 500")
print(result.level, result.action, [c.ticket_id for c in result.citations])
```

Veja [docs/AGENT.md](docs/AGENT.md) para o fluxo, os niveis e todas as opcoes.

## Analise de Atendimento

`support_engine.analytics` transforma tickets mascarados e os resultados do agente
em um unico relatorio: sentimento multilingue (ingles, portugues e espanhol,
substituindo o TextBlob, com backend LLM opcional), tendencias ao vivo que alertam
quando um tema cresce acima da sua linha de base, radar de risco para churn,
escalonamento e estouro de SLA, qualidade do atendimento por produto, candidatos a
causa raiz para tickets negativos e em alta, e deflexao com ROI do agente. Todos os
limiares, pesos e custos sao configuraveis.

```python
from support_engine.analytics import AnalyticsPipeline

report = AnalyticsPipeline().run(ingested.tickets, agent_results)
print(report.sentiment.negative_share, [a.keywords for a in report.alerts])
print([r.ticket_id for r in report.high_risk()], report.roi.deflection_rate)
```

Veja [docs/ANALYTICS.md](docs/ANALYTICS.md) para as formulas e todas as opcoes.

## Copiloto, API REST e MCP

`support_engine.copilot` ajuda o atendente em um ticket aberto: resposta sugerida
fundamentada no agente de atendimento (respostas curadas citadas, juiz e niveis),
resumo curto, proximo passo (enviar, revisar, escalar ou tratar manualmente, com
prioridade vinda do radar de risco) e orientacao de tom a partir do sentimento do
cliente. As mesmas operacoes saem por uma API REST e por um servidor MCP (stdio ou
HTTP em `/mcp`). A API exige um token de acesso em `SUPPORT_ENGINE_API_TOKEN` ou no
arquivo `.env` e nao inicia sem ele.

```python
from support_engine.copilot import Copilot

copilot = Copilot.from_tickets(history)
s = copilot.suggest(open_ticket)
print(s.reply, s.next_step.action, s.tone.tone, s.summary.text)
```

Veja [docs/COPILOT.md](docs/COPILOT.md) para os endpoints, as ferramentas MCP e as regras de seguranca.

## Integracao com outros projetos (NexLink)

`support_engine.nexlink` deixa os outros projetos do portfolio chamarem o Support
Engine com um envelope de mensagem unico (remetente, destino, acao, parametros, id
de correlacao): o NexHub dispara uma ingestao (`support.ingest`), o NexCoder le
metricas de suporte (`support.metrics`) e qualquer cliente com o escopo pede uma
resposta sugerida (`support.suggest_reply`). O Support Engine, por sua vez, pede ao
motor de relatorios do nexdash (askdash) o dashboard `support_metrics`. Cada projeto
chamador tem token e lista de escopos proprios, lidos do ambiente ou do `.env`.

```bash
python examples/nexlink/nexcoder_reads_metrics.py   # offline: NexCoder -> Support Engine -> nexdash
```

Veja [docs/NEXLINK.md](docs/NEXLINK.md) para o envelope, as acoes, os escopos e os exemplos.

## Linha de Comando (CLI)

```bash
# Importar tickets para o banco (PII mascarada por padrao; --no-mask-pii para desligar)
support-engine import --input tests/fixtures/tickets_sample.csv

# Limpar, mascarar, deduplicar e dividir em chunks; gravar chunks em JSON Lines
support-engine ingest --input tests/fixtures/tickets_sample.json --chunks-out chunks.jsonl

# Pontuar respostas, aplicar feedback e correcoes, relatar lacunas e metricas de qualidade
support-engine curate --input tests/fixtures/tickets_sample.json --feedback feedback.jsonl --report-out report.json

# Sentimento, tendencias, radar de risco, qualidade, causa raiz e ROI em um relatorio JSON
support-engine analyze --input tests/fixtures/tickets_sample.json --agent-results results.jsonl --report-out analytics.json

# API REST do copiloto e MCP via HTTP (exige o extra api e SUPPORT_ENGINE_API_TOKEN no ambiente ou no .env)
support-engine serve --input tests/fixtures/tickets_sample.json

# Servidor MCP do copiloto via stdio
support-engine mcp --input tests/fixtures/tickets_sample.json

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
