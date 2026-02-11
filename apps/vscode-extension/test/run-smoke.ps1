$ErrorActionPreference = 'Stop'

$testDir = $PSScriptRoot
$extDir = Split-Path -Parent $testDir

Push-Location $extDir
try {
  if (-not (Test-Path -Path 'node_modules')) {
    npm install
  }

  npm run build

  npx tsc `
    --target ES2020 `
    --module commonjs `
    --lib ES2020 `
    --esModuleInterop `
    --skipLibCheck `
    --types node `
    --outDir test/dist `
    --rootDir test `
    test/mock-server.ts test/smoke.ts

  $server = Start-Process -FilePath node -ArgumentList 'test/dist/mock-server.js' -PassThru
  try {
    Start-Sleep -Milliseconds 400
    node test/dist/smoke.js
    $exitCode = $LASTEXITCODE
  } finally {
    if ($null -ne $server -and -not $server.HasExited) {
      $server.Kill()
      $server.WaitForExit()
    }
  }

  exit $exitCode
} finally {
  Pop-Location
}
