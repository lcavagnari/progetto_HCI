$ErrorActionPreference = "Stop"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        winget install -e --id Python.Python.3.12
        Write-Warning "Python was just installed. Close this terminal, open a new one, and re-run this script."
        exit 1
    } else {
        Write-Error "winget not found. Install Python manually from https://www.python.org/downloads/"
        exit 1
    }
}

if (-not (Test-Path progetto_HCI)) {
    git clone https://github.com/lcavagnari/progetto_HCI.git
}
Set-Location progetto_HCI

if (-not (Test-Path venv)) {
    python -m venv venv
}
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

if (Test-Path ..\deap-dataset.zip) {
    Expand-Archive -Path ..\deap-dataset.zip -DestinationPath . -Force
} else {
    Write-Warning "..\deap-dataset.zip not found — skipping dataset extraction."
}

Write-Host ""
Write-Host "Done. In new terminals, activate the venv with:"
Write-Host "  cd $(Get-Location); .\venv\Scripts\Activate.ps1"
