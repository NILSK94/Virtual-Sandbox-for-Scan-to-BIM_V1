param([switch]$InstallLocalModel)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not $env:SCANBIM_PROVIDER -and -not $InstallLocalModel) {
    $env:SCANBIM_PROVIDER = 'openai'
}
$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host 'ScanBIM: Python-Umgebung wird eingerichtet ...'
    & py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 installieren: https://www.python.org/downloads/' }
}
$requirementsHash = (Get-FileHash -LiteralPath 'requirements.txt' -Algorithm SHA256).Hash
$installedMarker = Join-Path $PSScriptRoot '.venv\.requirements.sha256'
if (-not (Test-Path -LiteralPath $installedMarker) -or (Get-Content -LiteralPath $installedMarker -Raw).Trim() -ne $requirementsHash) {
    & $venvPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Paketinstallation fehlgeschlagen. Internetverbindung prüfen und erneut starten.' }
    Set-Content -LiteralPath $installedMarker -Value $requirementsHash
}

function Wait-LocalService($Url, $Process, $TimeoutSeconds = 25) {
    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    while ($timer.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
        if ($Process.HasExited) { return $false }
        & curl.exe --noproxy '*' --silent --fail --max-time 1 --output NUL $Url
        if ($LASTEXITCODE -eq 0) { return $true }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

function Ensure-DockerDesktop {
    if (-not (Test-Path -LiteralPath $dockerDesktop)) { return }
    if (Get-Process -Name 'Docker Desktop' -ErrorAction SilentlyContinue) { return }
    try {
        Write-Host 'ScanBIM: Docker Desktop wird fuer den Versuchsraum gestartet ...'
        Start-Process -FilePath $dockerDesktop -WindowStyle Hidden
    } catch {
        Write-Warning "Docker Desktop konnte nicht gestartet werden: $($_.Exception.Message)"
    }
}

$ollamaServer = $null
$studioServer = $null
$studioUrl = 'http://127.0.0.1:8765'
$dockerDesktop = Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe'
$pendingProvider = Join-Path $PSScriptRoot 'provider-next-start.txt'
if (Test-Path -LiteralPath $pendingProvider) {
    $requestedProvider = (Get-Content -LiteralPath $pendingProvider -Raw).Trim()
    if ($requestedProvider -notin @('openai', 'claude', 'ollama')) {
        throw 'Ungueltiger Anbieter fuer den naechsten Start.'
    }
    & $venvPython -m scanbim.settings --activate-provider $requestedProvider
    if ($LASTEXITCODE -ne 0) { throw 'Modellanbieter konnte nicht umgestellt werden. KI-Einstellungen pruefen.' }
    Remove-Item -LiteralPath $pendingProvider
}
$studioProvider = & $venvPython -c 'from scanbim.settings import active_provider; print(active_provider())'
if ($LASTEXITCODE -ne 0) { throw 'Gespeicherter Modellanbieter konnte nicht gelesen werden.' }
try {
    Ensure-DockerDesktop
    if ($studioProvider -eq 'ollama' -or $InstallLocalModel) {
        $ollama = Get-Command ollama.exe -ErrorAction SilentlyContinue
        $ollamaPath = if ($ollama) { $ollama.Source } else {
            Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe'
        }
        if (Test-Path -LiteralPath $ollamaPath) {
            $env:OLLAMA_HOST = '127.0.0.1:11434'
            $env:OLLAMA_NO_CLOUD = '1'
            $env:OLLAMA_NOPRUNE = '1'
            # Reuse the desktop Ollama instance, including its already loaded model.
            & curl.exe --noproxy '*' --silent --fail --max-time 2 --output NUL 'http://127.0.0.1:11434/api/version'
            $ollamaReady = $LASTEXITCODE -eq 0
            if (-not $ollamaReady) {
                $ollamaServer = Start-Process -FilePath $ollamaPath -ArgumentList 'serve' -WindowStyle Hidden -PassThru
                $ollamaReady = Wait-LocalService 'http://127.0.0.1:11434/api/version' $ollamaServer
            }
            if ($ollamaReady) {
                if ($InstallLocalModel) {
                    $model = & $venvPython -c 'from scanbim.ollama import DEFAULT_MODEL; print(DEFAULT_MODEL)'
                    & $ollamaPath pull $model
                    if ($LASTEXITCODE -ne 0) { throw 'Modelldownload fehlgeschlagen. Internetverbindung pruefen.' }
                }
            } elseif ($InstallLocalModel) {
                throw 'Lokales Ollama konnte nicht gestartet werden. Port 11434 pruefen.'
            } else {
                Write-Warning 'Lokales Ollama konnte nicht gestartet werden. Port 11434 pruefen.'
            }
        } elseif ($InstallLocalModel) {
            throw 'Ollama ist nicht installiert.'
        } else {
            Write-Warning 'Ollama ist nicht installiert. Der visuelle Agent benoetigt Ollama.'
        }
    }
    Write-Host 'ScanBIM Studio wird gestartet. Gespeicherte Szenen werden eingelesen ...'
    $studioServer = Start-Process -FilePath $venvPython -ArgumentList '-m uvicorn scanbim.app:app --host 127.0.0.1 --port 8765 --timeout-graceful-shutdown 3' -NoNewWindow -PassThru
    if (-not (Wait-LocalService $studioUrl $studioServer 90)) {
        if ($studioServer.HasExited) {
            throw "ScanBIM Studio wurde beim Start beendet (Exitcode $($studioServer.ExitCode)). Die Servermeldung steht oben."
        }
        throw "ScanBIM Studio antwortet nach 90 Sekunden nicht unter $studioUrl. Die Servermeldungen oben pruefen."
    }
    Write-Host "ScanBIM Studio: $studioUrl"
    Write-Host 'Dieses Fenster zum Betrieb offen lassen. Beenden mit Strg+C.'
    Start-Process $studioUrl
    while (-not $studioServer.WaitForExit(30000)) {
        Ensure-DockerDesktop
    }
} finally {
    foreach ($process in @($studioServer, $ollamaServer)) {
        if ($process -and -not $process.HasExited) { Stop-Process -InputObject $process -ErrorAction SilentlyContinue }
    }
}
