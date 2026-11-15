param(
    [Parameter(Position = 0)]
    [ValidateSet("local", "test", "docker", "docker-postgres", "help")]
    [string]$Mode = "local",

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Show-Usage {
    Write-Output "Usage: .\scripts\run.ps1 [local|test|docker|docker-postgres|help] [args...]"
    Write-Output ""
    Write-Output "Modes:"
    Write-Output "  local            Run support-engine locally in .venv (default, uses DB_URL or SQLite)"
    Write-Output "  test             Run pytest, ruff and check_no_emoji.py"
    Write-Output "  docker           Run via Docker Compose (SQLite default or DB_URL from .env)"
    Write-Output "  docker-postgres  Run via Docker Compose with the optional PostgreSQL service"
    Write-Output "  help             Show this message"
}

function Get-VenvPython {
    $VenvPy = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path $VenvPy)) {
        python -m venv .venv
        & $VenvPy -m pip install --upgrade pip | Out-Host
        & $VenvPy -m pip install -e ".[dev]" | Out-Host
    }
    return $VenvPy
}

switch ($Mode) {
    "help" {
        Show-Usage
        exit 0
    }
    "local" {
        $Py = Get-VenvPython
        if (-not $ExtraArgs -or $ExtraArgs.Count -eq 0) {
            & $Py -m support_engine.cli --help
        } else {
            & $Py -m support_engine.cli @ExtraArgs
        }
    }
    "test" {
        $Py = Get-VenvPython
        & $Py -m pytest @ExtraArgs
        & $Py -m ruff check .
        & $Py scripts/check_no_emoji.py
    }
    "docker" {
        docker compose up --build @ExtraArgs
    }
    "docker-postgres" {
        if (-not $env:DB_URL) {
            $env:DB_URL = "postgresql+psycopg://support:changeme@postgres:5432/support_engine"
        }
        docker compose --profile postgres up --build @ExtraArgs
    }
}
