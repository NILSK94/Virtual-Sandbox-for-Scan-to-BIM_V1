$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not $env:SCANBIM_PROVIDER) {
    $env:SCANBIM_PROVIDER = 'openai'
}
$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) { throw 'Bitte zuerst Start.bat ausfuehren.' }
$settingsFile = & $venvPython -m scanbim.settings --init
if ($LASTEXITCODE -ne 0) { throw 'Einstellungsdatei konnte nicht vorbereitet werden.' }
Start-Process -FilePath 'notepad.exe' -ArgumentList ('"' + $settingsFile + '"')
