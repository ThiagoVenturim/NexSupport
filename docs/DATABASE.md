# Database and infrastructure guide

Support Intelligence Engine provides a pluggable storage layer built on SQLAlchemy Core
with fully parameterized SQL queries, zero-configuration local SQLite as default, and
production support for external databases via `DB_URL`.

## Default: SQLite (zero configuration)

When `DB_URL` is not set in the environment or in a local `.env` file, Support Intelligence
Engine defaults to a local SQLite database at `support_engine.db`:

```
sqlite:///support_engine.db
```

No external database server or daemon is required to get started.

## Custom database via DB_URL

To connect Support Intelligence Engine to an external database, configure the `DB_URL`
variable in your shell environment or in a local `.env` file. You can start from the provided
`.env.example`:

```bash
cp .env.example .env
```

The database URL follows standard SQLAlchemy URL formats.

### PostgreSQL

Install the optional PostgreSQL driver:

```bash
pip install -e ".[postgres]"
```

Configure `DB_URL`:

```env
DB_URL=postgresql+psycopg://username:password@localhost:5432/support_engine
```

### MySQL / MariaDB

Install the optional MySQL driver:

```bash
pip install -e ".[mysql]"
```

Configure `DB_URL`:

```env
DB_URL=mysql+pymysql://username:password@localhost:3306/support_engine
```

### Microsoft SQL Server (ODBC)

Install the optional ODBC driver wrapper:

```bash
pip install -e ".[mssql]"
```

Configure `DB_URL` with your system's installed ODBC driver:

```env
DB_URL=mssql+pyodbc://username:password@localhost:1433/support_engine?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes
```

## Security and PII masking

All database queries use SQLAlchemy parameter binding (`:param` syntax) to prevent SQL injection.
In addition, Support Intelligence Engine includes a built-in PII anonymization engine
(`support_engine.pii`) capable of masking:

- Email addresses (`[EMAIL_n]`)
- Credit card numbers (`[CARD_n]`)
- Brazilian CPF (`[CPF_n]`) and CNPJ (`[CNPJ_n]`)
- Phone numbers (`[PHONE_n]`)
- IPv4 addresses (`[IP_n]`)
- API keys and tokens (`[SECRET_n]`)

To mask sensitive entities before storing them in the database:

```bash
support-engine import --input tickets.csv --mask-pii
```

Or programmatically in Python:

```python
from support_engine import TicketStore, mask_ticket

with TicketStore() as store:
    store.save_ticket(ticket, mask_pii=True)
```

## Running with Docker and Docker Compose

A `Dockerfile` and `docker-compose.yml` are provided at the repository root.

- By default, `docker compose up` runs Support Intelligence Engine using local container storage (`/data/support_engine.db`):

```bash
docker compose up
```

- To start a bundled PostgreSQL database alongside the application, use the `postgres` profile:

```bash
docker compose --profile postgres up
```

## Running on Kubernetes

Example manifests are available in `k8s/`:

- `k8s/deployment.yaml`: Deployment spec with non-root security context, resource limits, and persistent volume mounts.
- `k8s/service.yaml`: ClusterIP service.
- `k8s/configmap.yaml`: Non-sensitive configuration (`LOG_LEVEL`).
- `k8s/secret.yaml`: Sensitive credentials (`DB_URL`).

Deploy with kubectl:

```bash
kubectl apply -f k8s/
```

## Multiplatform execution scripts

Helper scripts are available in `scripts/`:

- Windows (PowerShell): `.\scripts\run.ps1 [local|test|docker|docker-postgres|help]`
- Linux / macOS (Bash): `./scripts/run.sh [local|test|docker|docker-postgres|help]`
