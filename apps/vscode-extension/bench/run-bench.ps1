param(
  [int]$Requests = 50,
  [string]$Sizes = "short,medium,long",
  [string]$Languages = "typescript,python"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ExtensionRoot = Resolve-Path (Join-Path $ScriptDir "..")
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..\..")
$ResultsDir = Join-Path $ScriptDir "results"
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$Port = if ($env:LLAMA_SERVER_PORT) { [int]$env:LLAMA_SERVER_PORT } else { 8787 }
$Url = "http://127.0.0.1:$Port/v1/completions"

New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null

$JsonPath = Join-Path $ResultsDir "bench-$Timestamp.json"
$TablePath = Join-Path $ResultsDir "bench-$Timestamp.txt"
$LogPath = Join-Path $ResultsDir "llama-server-$Timestamp.log"

function Test-PortOpen {
  param(
    [string]$Host,
    [int]$Port
  )
  return Test-NetConnection -ComputerName $Host -Port $Port -InformationLevel Quiet
}

if (-not (Test-PortOpen -Host "127.0.0.1" -Port $Port)) {
  Write-Host "llama-server not detected on port $Port. Starting..."

  $ServerPath = $env:LLAMA_SERVER_PATH
  if (-not $ServerPath) {
    $Candidate = Join-Path $RepoRoot "llama-server.exe"
    if (Test-Path $Candidate) {
      $ServerPath = $Candidate
    } else {
      $Command = Get-Command "llama-server.exe" -ErrorAction SilentlyContinue
      if ($Command) {
        $ServerPath = $Command.Source
      }
    }
  }

  if (-not $ServerPath) {
    throw "llama-server.exe not found. Set LLAMA_SERVER_PATH or add it to PATH."
  }

  $ModelPath = $env:LLAMA_MODEL_PATH
  if (-not $ModelPath) {
    $DefaultModel = Join-Path $RepoRoot "models\Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
    if (Test-Path $DefaultModel) {
      $ModelPath = $DefaultModel
    }
  }

  if (-not $ModelPath -or -not (Test-Path $ModelPath)) {
    throw "Model not found. Set LLAMA_MODEL_PATH to your GGUF model path."
  }

  $Args = @(
    "--model", $ModelPath,
    "--host", "127.0.0.1",
    "--port", "$Port",
    "--ctx-size", "4096"
  )

  if ($env:LLAMA_SERVER_ARGS) {
    $Extra = $env:LLAMA_SERVER_ARGS.Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries)
    $Args += $Extra
  }

  Start-Process -FilePath $ServerPath -ArgumentList $Args -WorkingDirectory $RepoRoot -RedirectStandardOutput $LogPath -RedirectStandardError $LogPath -WindowStyle Hidden | Out-Null

  $Started = $false
  for ($i = 0; $i -lt 30; $i += 1) {
    Start-Sleep -Seconds 1
    if (Test-PortOpen -Host "127.0.0.1" -Port $Port) {
      $Started = $true
      break
    }
  }

  if (-not $Started) {
    throw "llama-server failed to start within 30s. Check $LogPath"
  }
}

Push-Location $ExtensionRoot
try {
  & npx --yes tsc bench/benchmark.ts --outDir bench/dist --module commonjs --target es2020 --lib es2020,dom --moduleResolution node --esModuleInterop
  & node bench/dist/benchmark.js --requests $Requests --url $Url --sizes $Sizes --languages $Languages --output $JsonPath --table $TablePath
  Write-Host "Saved JSON results to $JsonPath"
  Write-Host "Saved table results to $TablePath"
} finally {
  Pop-Location
}
