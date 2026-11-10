$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
docker info --format '{{.ServerVersion}}'
if ($LASTEXITCODE -ne 0) { throw 'Bitte Docker Desktop mit Linux-Containern starten.' }
docker build --tag scanbim-experiment:2 sandbox
if ($LASTEXITCODE -ne 0) { throw 'Experiment-Image konnte nicht gebaut werden.' }
Write-Host 'Versuchsraum bereit. ScanBIM im Browser neu laden.'
