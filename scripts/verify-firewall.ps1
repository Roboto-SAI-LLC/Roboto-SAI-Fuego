#Requires -RunAsAdministrator
<#
.SYNOPSIS
  Verifies Roboto SAI firewall rules and loopback-only connectivity.

.DESCRIPTION
  Confirms expected firewall rules exist, checks llama-server loopback connectivity,
  and validates that external network access is blocked from the Code.exe context.

.EXAMPLE
  .\verify-firewall.ps1

.EXAMPLE
  .\verify-firewall.ps1 -ExternalUrl "https://example.com/"

.EXAMPLE
  .\verify-firewall.ps1 -CodePath "C:\Program Files\Microsoft VS Code\Code.exe"
#>
[CmdletBinding()]
param(
  [string]$CodePath,
  [string]$CodeInsidersPath,
  [string]$LlamaServerPath,
  [string]$NodePath,
  [string]$GitPath,
  [string]$PythonPath,
  [string]$PowerShellPath,
  [string]$ExternalUrl = 'https://marketplace.visualstudio.com',
  [int]$LlamaPort = 8787
)

$ErrorActionPreference = 'Stop'

$RulePrefix = 'Roboto SAI'
$LoopbackAddresses = @('127.0.0.1', '::1')
$NonLoopbackRanges = @(
  '0.0.0.0-126.255.255.255',
  '127.0.0.2-127.255.255.255',
  '128.0.0.0-255.255.255.255',
  '::2-ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff'
)

function Write-Status {
  param(
    [string]$Message,
    [string]$Level = 'INFO'
  )

  Write-Host ("[{0}] {1}" -f $Level, $Message)
}

function Fail {
  param([string]$Message)

  Write-Error $Message
  exit 1
}

function Resolve-CandidatePath {
  param([string]$Candidate)

  if (-not $Candidate) {
    return $null
  }

  if ($Candidate -match '[\*\?]') {
    $match = Get-ChildItem -Path $Candidate -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($match) {
      return $match.FullName
    }
  } elseif (Test-Path -Path $Candidate) {
    return (Resolve-Path -Path $Candidate).Path
  }

  return $null
}

function Resolve-ExecutablePath {
  param(
    [string]$ProvidedPath,
    [string[]]$CandidatePaths,
    [string[]]$CommandNames
  )

  if ($ProvidedPath) {
    if (Test-Path -Path $ProvidedPath) {
      return (Resolve-Path -Path $ProvidedPath).Path
    }
    Write-Status "Provided path not found: $ProvidedPath" 'WARN'
  }

  foreach ($commandName in $CommandNames) {
    if (-not $commandName) {
      continue
    }
    $command = Get-Command -Name $commandName -CommandType Application -ErrorAction SilentlyContinue
    if ($command -and $command.Source) {
      return (Resolve-Path -Path $command.Source).Path
    }
  }

  foreach ($candidate in $CandidatePaths) {
    $resolved = Resolve-CandidatePath -Candidate $candidate
    if ($resolved) {
      return $resolved
    }
  }

  return $null
}

function Get-RuleName {
  param(
    [string]$ProgramLabel,
    [string]$Direction,
    [string]$Action
  )

  return "$RulePrefix $Action $Direction - $ProgramLabel"
}

function Test-AddressSubset {
  param(
    [string[]]$Actual,
    [string[]]$Required
  )

  if (-not $Required -or $Required.Count -eq 0) {
    return $true
  }

  $flattened = @()
  foreach ($item in $Actual) {
    if ($item) {
      $flattened += $item -split ','
    }
  }

  $actualSet = $flattened | ForEach-Object { $_.Trim() } | Where-Object { $_ }

  foreach ($requiredItem in $Required) {
    if (-not ($actualSet -contains $requiredItem)) {
      return $false
    }
  }

  return $true
}

function Assert-Rule {
  param(
    [string]$DisplayName,
    [string]$Direction,
    [string]$Action,
    [string]$Program,
    [string[]]$RemoteAddress,
    [string[]]$LocalAddress
  )

  $rule = Get-NetFirewallRule -DisplayName $DisplayName -ErrorAction SilentlyContinue
  if (-not $rule) {
    Write-Status "Missing rule: $DisplayName" 'ERROR'
    return $false
  }

  if ($rule.Direction -ne $Direction -or $rule.Action -ne $Action) {
    Write-Status "Rule mismatch: $DisplayName" 'ERROR'
    return $false
  }

  $appFilter = Get-NetFirewallApplicationFilter -AssociatedNetFirewallRule $rule
  if ($appFilter.Program -ne $Program) {
    Write-Status "Program mismatch for rule: $DisplayName" 'ERROR'
    return $false
  }

  $addressFilter = Get-NetFirewallAddressFilter -AssociatedNetFirewallRule $rule
  if (-not (Test-AddressSubset -Actual $addressFilter.RemoteAddress -Required $RemoteAddress)) {
    Write-Status "Remote address mismatch for rule: $DisplayName" 'ERROR'
    return $false
  }

  if (-not (Test-AddressSubset -Actual $addressFilter.LocalAddress -Required $LocalAddress)) {
    Write-Status "Local address mismatch for rule: $DisplayName" 'ERROR'
    return $false
  }

  return $true
}

function Test-LlamaLoopback {
  param([int]$Port)

  $probe = Test-NetConnection -ComputerName '127.0.0.1' -Port $Port -InformationLevel Detailed
  if (-not $probe.TcpTestSucceeded) {
    Write-Status "localhost:$Port is not reachable." 'ERROR'
    return $false
  }

  return $true
}

function Test-CodeExternalBlocked {
  param(
    [string]$CodeExecutable,
    [string]$Url
  )

  $tempRoot = Join-Path -Path $env:TEMP -ChildPath ("RobotoSAI-FirewallTest-" + [Guid]::NewGuid().ToString('N'))
  $userData = Join-Path -Path $tempRoot -ChildPath 'user-data'
  $extDir = Join-Path -Path $tempRoot -ChildPath 'extensions'
  $stdout = Join-Path -Path $tempRoot -ChildPath 'stdout.txt'
  $stderr = Join-Path -Path $tempRoot -ChildPath 'stderr.txt'

  New-Item -Path $userData -ItemType Directory -Force | Out-Null
  New-Item -Path $extDir -ItemType Directory -Force | Out-Null

  $arguments = @(
    '--user-data-dir', $userData,
    '--extensions-dir', $extDir,
    '--install-extension', 'ms-vscode.powershell',
    '--force'
  )

  Write-Status "Attempting external access to $Url via Code.exe extension install." 'INFO'

  $process = Start-Process -FilePath $CodeExecutable -ArgumentList $arguments -PassThru -Wait -NoNewWindow -RedirectStandardOutput $stdout -RedirectStandardError $stderr
  $output = ''
  if (Test-Path -Path $stdout) {
    $output += Get-Content -Path $stdout -Raw -ErrorAction SilentlyContinue
  }
  if (Test-Path -Path $stderr) {
    $output += Get-Content -Path $stderr -Raw -ErrorAction SilentlyContinue
  }

  Remove-Item -Path $tempRoot -Recurse -Force -ErrorAction SilentlyContinue

  if ($process.ExitCode -eq 0) {
    Write-Status 'External access succeeded from Code.exe context (unexpected).' 'ERROR'
    return $false
  }

  if ($output -match '(ENOTFOUND|ECONNREFUSED|ETIMEDOUT|getaddrinfo|Network is unreachable|blocked by the firewall)') {
    return $true
  }

  Write-Status 'External access test failed for an unexpected reason.' 'ERROR'
  if ($output) {
    Write-Status $output.Trim() 'ERROR'
  }
  return $false
}

$repoRoot = Resolve-Path -Path (Join-Path -Path $PSScriptRoot -ChildPath '..')

$codeCandidates = @(
  (Join-Path -Path $Env:LocalAppData -ChildPath 'Programs\Microsoft VS Code\Code.exe'),
  (Join-Path -Path $Env:ProgramFiles -ChildPath 'Microsoft VS Code\Code.exe'),
  (Join-Path -Path ${Env:ProgramFiles(x86)} -ChildPath 'Microsoft VS Code\Code.exe')
)

$codeInsidersCandidates = @(
  (Join-Path -Path $Env:LocalAppData -ChildPath 'Programs\Microsoft VS Code Insiders\Code - Insiders.exe'),
  (Join-Path -Path $Env:ProgramFiles -ChildPath 'Microsoft VS Code Insiders\Code - Insiders.exe'),
  (Join-Path -Path ${Env:ProgramFiles(x86)} -ChildPath 'Microsoft VS Code Insiders\Code - Insiders.exe')
)

$llamaCandidates = @(
  (Join-Path -Path $repoRoot -ChildPath 'apps\model-training\tools\llama-bin\llama-server.exe')
)

$nodeCandidates = @(
  (Join-Path -Path $Env:ProgramFiles -ChildPath 'nodejs\node.exe'),
  (Join-Path -Path ${Env:ProgramFiles(x86)} -ChildPath 'nodejs\node.exe'),
  (Join-Path -Path $Env:LocalAppData -ChildPath 'Programs\nodejs\node.exe')
)

$gitCandidates = @(
  (Join-Path -Path $Env:ProgramFiles -ChildPath 'Git\cmd\git.exe'),
  (Join-Path -Path $Env:ProgramFiles -ChildPath 'Git\bin\git.exe'),
  (Join-Path -Path ${Env:ProgramFiles(x86)} -ChildPath 'Git\cmd\git.exe'),
  (Join-Path -Path ${Env:ProgramFiles(x86)} -ChildPath 'Git\bin\git.exe')
)

$pythonCandidates = @(
  (Join-Path -Path $Env:LocalAppData -ChildPath 'Programs\Python\Python*\python.exe'),
  'C:\Python*\python.exe'
)

$powerShellCandidates = @(
  (Join-Path -Path $Env:SystemRoot -ChildPath 'System32\WindowsPowerShell\v1.0\powershell.exe')
)

$resolvedCodePath = Resolve-ExecutablePath -ProvidedPath $CodePath -CandidatePaths $codeCandidates -CommandNames @('code')
$resolvedCodeInsidersPath = Resolve-ExecutablePath -ProvidedPath $CodeInsidersPath -CandidatePaths $codeInsidersCandidates -CommandNames @('code-insiders')
$resolvedLlamaPath = Resolve-ExecutablePath -ProvidedPath $LlamaServerPath -CandidatePaths $llamaCandidates -CommandNames @('llama-server.exe')
$resolvedNodePath = Resolve-ExecutablePath -ProvidedPath $NodePath -CandidatePaths $nodeCandidates -CommandNames @('node')
$resolvedGitPath = Resolve-ExecutablePath -ProvidedPath $GitPath -CandidatePaths $gitCandidates -CommandNames @('git')
$resolvedPythonPath = Resolve-ExecutablePath -ProvidedPath $PythonPath -CandidatePaths $pythonCandidates -CommandNames @('python', 'python3')
$resolvedPowerShellPath = Resolve-ExecutablePath -ProvidedPath $PowerShellPath -CandidatePaths $powerShellCandidates -CommandNames @('powershell', 'powershell.exe')

$missingPrograms = @()
if (-not $resolvedCodePath -and -not $resolvedCodeInsidersPath) {
  $missingPrograms += 'Code.exe or Code - Insiders.exe'
}
if (-not $resolvedLlamaPath) {
  $missingPrograms += 'llama-server.exe'
}
if (-not $resolvedNodePath) {
  $missingPrograms += 'node.exe'
}
if (-not $resolvedGitPath) {
  $missingPrograms += 'git.exe'
}
if (-not $resolvedPythonPath) {
  $missingPrograms += 'python.exe'
}
if (-not $resolvedPowerShellPath) {
  $missingPrograms += 'powershell.exe'
}

if ($missingPrograms.Count -gt 0) {
  Fail "Required executable(s) not found: $($missingPrograms -join ', '). Provide explicit paths or install the missing tools."
}

$programEntries = @()
if ($resolvedCodePath) {
  $programEntries += [pscustomobject]@{ Label = 'Code.exe'; Path = $resolvedCodePath }
}
if ($resolvedCodeInsidersPath) {
  $programEntries += [pscustomobject]@{ Label = 'Code - Insiders.exe'; Path = $resolvedCodeInsidersPath }
}
if ($resolvedLlamaPath) {
  $programEntries += [pscustomobject]@{ Label = 'llama-server.exe'; Path = $resolvedLlamaPath }
}
$programEntries += [pscustomobject]@{ Label = 'node.exe'; Path = $resolvedNodePath }
$programEntries += [pscustomobject]@{ Label = 'git.exe'; Path = $resolvedGitPath }
$programEntries += [pscustomobject]@{ Label = 'python.exe'; Path = $resolvedPythonPath }
$programEntries += [pscustomobject]@{ Label = 'powershell.exe'; Path = $resolvedPowerShellPath }

$ruleFailures = 0
foreach ($program in $programEntries) {
  $allowInbound = Get-RuleName -ProgramLabel $program.Label -Direction 'Inbound' -Action 'Allow Loopback'
  $blockInbound = Get-RuleName -ProgramLabel $program.Label -Direction 'Inbound' -Action 'Block External'
  $allowOutbound = Get-RuleName -ProgramLabel $program.Label -Direction 'Outbound' -Action 'Allow Loopback'
  $blockOutbound = Get-RuleName -ProgramLabel $program.Label -Direction 'Outbound' -Action 'Block External'

  if (-not (Assert-Rule -DisplayName $allowInbound -Direction 'Inbound' -Action 'Allow' -Program $program.Path -RemoteAddress $LoopbackAddresses -LocalAddress $LoopbackAddresses)) { $ruleFailures++ }
  if (-not (Assert-Rule -DisplayName $blockInbound -Direction 'Inbound' -Action 'Block' -Program $program.Path -RemoteAddress $NonLoopbackRanges -LocalAddress @('Any'))) { $ruleFailures++ }
  if (-not (Assert-Rule -DisplayName $allowOutbound -Direction 'Outbound' -Action 'Allow' -Program $program.Path -RemoteAddress $LoopbackAddresses -LocalAddress @('Any'))) { $ruleFailures++ }
  if (-not (Assert-Rule -DisplayName $blockOutbound -Direction 'Outbound' -Action 'Block' -Program $program.Path -RemoteAddress $NonLoopbackRanges -LocalAddress @('Any'))) { $ruleFailures++ }
}

if ($ruleFailures -gt 0) {
  Fail "Firewall rule verification failed with $ruleFailures error(s)."
}

if (-not (Test-LlamaLoopback -Port $LlamaPort)) {
  Fail 'Loopback test failed.'
}

$codeUnderTest = if ($resolvedCodePath) { $resolvedCodePath } else { $resolvedCodeInsidersPath }
if (-not (Test-CodeExternalBlocked -CodeExecutable $codeUnderTest -Url $ExternalUrl)) {
  Fail 'External access test failed.'
}

Write-Status 'Firewall verification passed.' 'INFO'
