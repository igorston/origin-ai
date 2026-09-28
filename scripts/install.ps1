<#
.SYNOPSIS
  Installs Origin on Windows: a Python virtual environment, the dependencies, a .env
  file and the Ollama models.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\install.ps1
  powershell -ExecutionPolicy Bypass -File scripts\install.ps1 -SkipModels
#>
param(
    [switch]$SkipModels  # download the models later (ollama pull, or the first-run screen)
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

function Step($text) { Write-Host "`n==> $text" -ForegroundColor Cyan }

Step "Python 3.11+"
$python = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $python) { throw "Python not found. Install Python 3.11 or newer from https://www.python.org/downloads/" }
$version = & $python -c "import sys; print('%d.%d' % sys.version_info[:2])"
if ([version]$version -lt [version]"3.11") { throw "Python $version found; Origin needs 3.11 or newer." }
Write-Host "Python $version ($python)"

Step "Virtual environment and dependencies"
if (-not (Test-Path .venv)) { & $python -m venv .venv }
.\.venv\Scripts\python.exe -m pip install --quiet --upgrade pip
.\.venv\Scripts\python.exe -m pip install --quiet -e .

Step "Configuration"
if (Test-Path .env) {
    Write-Host ".env already exists: kept as is."
} else {
    Copy-Item .env.example .env
    Write-Host "Created .env from .env.example (edit it to change models, language, etc.)."
}

function Setting($name, $default) {
    $line = Select-String -Path .env -Pattern "^\s*$name\s*=\s*(.+)$" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($line) { return $line.Matches[0].Groups[1].Value.Trim() } else { return $default }
}

Step "Ollama"
$ollama = (Get-Command ollama -ErrorAction SilentlyContinue).Source
if (-not $ollama) {
    Write-Warning "Ollama not found. Install it from https://ollama.com/download; the first-run screen will then download the models."
} elseif ($SkipModels) {
    Write-Host "Skipping model downloads (-SkipModels)."
} else {
    foreach ($model in @((Setting "OLLAMA_MODEL" "qwen3:8b"), (Setting "OLLAMA_EMBED_MODEL" "bge-m3"))) {
        Write-Host "ollama pull $model"
        & $ollama pull $model
    }
}

Step "Done"
Write-Host "Start it with:  .\.venv\Scripts\python.exe main.py"
Write-Host "Then open:      http://127.0.0.1:$(Setting 'ORIGIN_PORT' '8000')"
Write-Host "White label:    copy brand\brand.example.json to brand\brand.json and edit it."
