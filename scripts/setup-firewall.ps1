#Requires -RunAsAdministrator
<#!
.SYNOPSIS
  Configures Roboto SAI firewall rules to allow loopback-only traffic and block all external traffic.

.DESCRIPTION
  Creates inbound and outbound rules for Code.exe, Code - Insiders.exe, llama-server.exe, and common
  VS Code child processes. Rules allow only 127.0.0.1/::1 and block all non-loopback addresses.

.EXAMPLE
  .\setup-firewall.ps1

.EXAMPLE
  .\setup-firewall.ps1 -Rollback

.EXAMPLE
  .\setup-firewall.ps1 -Audit

.EXAMPLE
  .\setup-firewall.ps1 -CodePath "C:\Program Files\Microsoft VS Code\Code.exe" -LlamaServerPath "R:\Repos\Roboto-SAI-2026\apps\model-training\tools\llama-bin\llama-server.exe"
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
  [switch]$Rollback,
  [switch]$Audit
)

$ErrorActionPreference = 'Stop'

$RulePrefix = 'Roboto SAI'
$RuleGroup = 'Roboto SAI'
$LoopbackAddresses = @('127.0.0.1', '::1')
$NonLoopbackRanges = @(
  '0.0.0.0-126.255.255.255',
  '127.0.0.2-127.255.255.255',
  '128.0.0.0-255.255.255.255',
  '::2-ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff'
)

$script:ExpectedRules = @()

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

function Get-RobotoRules {
  $rules = @()
  $rules += Get-NetFirewallRule -Group $RuleGroup -ErrorAction SilentlyContinue
  $rules += Get-NetFirewallRule -DisplayName "$RulePrefix *" -ErrorAction SilentlyContinue

  $unique = @{}
  foreach ($rule in $rules) {
    if ($rule) {
      $unique[$rule.Name] = $rule
    }
  }

  return $unique.Values
}

function Remove-RobotoRules {
  $rules = Get-RobotoRules
  if ($rules.Count -gt 0) {
    $rules | Remove-NetFirewallRule | Out-Null
    Write-Status "Removed $($rules.Count) Roboto SAI firewall rule(s)." 'INFO'
  }
}

function Add-ExpectedRule {
  param(
    [string]$DisplayName,
    [string]$Direction,
    [string]$Action,
    [string]$Program,
    [string[]]$RemoteAddress,
    [string[]]$LocalAddress
  )

  $script:ExpectedRules += [pscustomobject]@{
    DisplayName = $DisplayName
    Direction = $Direction
    Action = $Action
    Program = $Program
    RemoteAddress = $RemoteAddress
    LocalAddress = $LocalAddress
  }
}

function New-RobotoRule {
  param(
    [string]$DisplayName,
    [string]$Direction,
    [string]$Action,
    [string]$Program,
    [string[]]$RemoteAddress,
    [string[]]$LocalAddress
  )

  $params = @{
    DisplayName = $DisplayName
    Group = $RuleGroup
    Direction = $Direction
    Action = $Action
    Program = $Program
    Profile = 'Any'
    Protocol = 'Any'
  }

  if ($RemoteAddress) {
    $params.RemoteAddress = ($RemoteAddress -join ',')
  }

  if ($LocalAddress) {
    $params.LocalAddress = ($LocalAddress -join ',')
  }

  New-NetFirewallRule @params | Out-Null
}

function Get-RuleName {
  param(
    [string]$ProgramLabel,
    [string]$Direction,
    [string]$Action
  )

  return "$RulePrefix $Action $Direction - $ProgramLabel"
}

function Apply-ProgramRules {
  param(
    [string]$ProgramLabel,
    [string]$ProgramPath
  )

  $allowInbound = Get-RuleName -ProgramLabel $ProgramLabel -Direction 'Inbound' -Action 'Allow Loopback'
  $blockInbound = Get-RuleName -ProgramLabel $ProgramLabel -Direction 'Inbound' -Action 'Block External'
  $allowOutbound = Get-RuleName -ProgramLabel $ProgramLabel -Direction 'Outbound' -Action 'Allow Loopback'
  $blockOutbound = Get-RuleName -ProgramLabel $ProgramLabel -Direction 'Outbound' -Action 'Block External'

  New-RobotoRule -DisplayName $allowInbound -Direction 'Inbound' -Action 'Allow' -Program $ProgramPath -RemoteAddress $LoopbackAddresses -LocalAddress $LoopbackAddresses
  Add-ExpectedRule -DisplayName $allowInbound -Direction 'Inbound' -Action 'Allow' -Program $ProgramPath -RemoteAddress $LoopbackAddresses -LocalAddress $LoopbackAddresses

  New-RobotoRule -DisplayName $blockInbound -Direction 'Inbound' -Action 'Block' -Program $ProgramPath -RemoteAddress $NonLoopbackRanges -LocalAddress @('Any')
  Add-ExpectedRule -DisplayName $blockInbound -Direction 'Inbound' -Action 'Block' -Program $ProgramPath -RemoteAddress $NonLoopbackRanges -LocalAddress @('Any')

  New-RobotoRule -DisplayName $allowOutbound -Direction 'Outbound' -Action 'Allow' -Program $ProgramPath -RemoteAddress $LoopbackAddresses -LocalAddress @('Any')
  Add-ExpectedRule -DisplayName $allowOutbound -Direction 'Outbound' -Action 'Allow' -Program $ProgramPath -RemoteAddress $LoopbackAddresses -LocalAddress @('Any')

  New-RobotoRule -DisplayName $blockOutbound -Direction 'Outbound' -Action 'Block' -Program $ProgramPath -RemoteAddress $NonLoopbackRanges -LocalAddress @('Any')
  Add-ExpectedRule -DisplayName $blockOutbound -Direction 'Outbound' -Action 'Block' -Program $ProgramPath -RemoteAddress $NonLoopbackRanges -LocalAddress @('Any')
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

function Test-ExpectedRule {
  param(
    [pscustomobject]$Spec
  )

  $rule = Get-NetFirewallRule -DisplayName $Spec.DisplayName -ErrorAction SilentlyContinue
  if (-not $rule) {
    Write-Status "Missing rule: $($Spec.DisplayName)" 'ERROR'
    return $false
  }

  if ($rule.Direction -ne $Spec.Direction -or $rule.Action -ne $Spec.Action) {
    Write-Status "Rule mismatch: $($Spec.DisplayName)" 'ERROR'
    return $false
  }

  $appFilter = Get-NetFirewallApplicationFilter -AssociatedNetFirewallRule $rule
  if ($appFilter.Program -ne $Spec.Program) {
    Write-Status "Program mismatch for rule: $($Spec.DisplayName)" 'ERROR'
    return $false
  }

  $addressFilter = Get-NetFirewallAddressFilter -AssociatedNetFirewallRule $rule
  if (-not (Test-AddressSubset -Actual $addressFilter.RemoteAddress -Required $Spec.RemoteAddress)) {
    Write-Status "Remote address mismatch for rule: $($Spec.DisplayName)" 'ERROR'
    return $false
  }

  if (-not (Test-AddressSubset -Actual $addressFilter.LocalAddress -Required $Spec.LocalAddress)) {
    Write-Status "Local address mismatch for rule: $($Spec.DisplayName)" 'ERROR'
    return $false
  }

  return $true
}

function Verify-ExpectedRules {
  $failures = 0
  foreach ($spec in $script:ExpectedRules) {
    if (-not (Test-ExpectedRule -Spec $spec)) {
      $failures++
    }
  }

  if ($failures -gt 0) {
    Fail "Firewall rule verification failed with $failures error(s)."
  }
}

if ($Rollback -and $Audit) {
  Fail 'Cannot use -Rollback and -Audit together.'
}

if ($Audit) {
  $rules = Get-RobotoRules | Sort-Object DisplayName
  if ($rules.Count -eq 0) {
    Write-Status 'No Roboto SAI firewall rules found.' 'INFO'
    return
  }

  $rules | ForEach-Object {
    $appFilter = Get-NetFirewallApplicationFilter -AssociatedNetFirewallRule $_
    $addressFilter = Get-NetFirewallAddressFilter -AssociatedNetFirewallRule $_
    [pscustomobject]@{
      DisplayName = $_.DisplayName
      Direction = $_.Direction
      Action = $_.Action
      Program = $appFilter.Program
      RemoteAddress = ($addressFilter.RemoteAddress -join ',')
      LocalAddress = ($addressFilter.LocalAddress -join ',')
    }
  }

  return
}

if ($Rollback) {
  Remove-RobotoRules
  Write-Status 'Rollback complete.' 'INFO'
  return
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

Write-Status "Using Code.exe: $resolvedCodePath" 'INFO'
Write-Status "Using Code - Insiders.exe: $resolvedCodeInsidersPath" 'INFO'
Write-Status "Using llama-server.exe: $resolvedLlamaPath" 'INFO'
Write-Status "Using node.exe: $resolvedNodePath" 'INFO'
Write-Status "Using git.exe: $resolvedGitPath" 'INFO'
Write-Status "Using python.exe: $resolvedPythonPath" 'INFO'
Write-Status "Using powershell.exe: $resolvedPowerShellPath" 'INFO'

Remove-RobotoRules

if ($resolvedCodePath) {
  Apply-ProgramRules -ProgramLabel 'Code.exe' -ProgramPath $resolvedCodePath
}

if ($resolvedCodeInsidersPath) {
  Apply-ProgramRules -ProgramLabel 'Code - Insiders.exe' -ProgramPath $resolvedCodeInsidersPath
}

Apply-ProgramRules -ProgramLabel 'llama-server.exe' -ProgramPath $resolvedLlamaPath
Apply-ProgramRules -ProgramLabel 'node.exe' -ProgramPath $resolvedNodePath
Apply-ProgramRules -ProgramLabel 'git.exe' -ProgramPath $resolvedGitPath
Apply-ProgramRules -ProgramLabel 'python.exe' -ProgramPath $resolvedPythonPath
Apply-ProgramRules -ProgramLabel 'powershell.exe' -ProgramPath $resolvedPowerShellPath

Verify-ExpectedRules

Write-Status 'Firewall configuration complete.' 'INFO'
