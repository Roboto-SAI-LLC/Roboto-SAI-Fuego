[CmdletBinding()]
param(
	[Parameter(ValueFromRemainingArguments = $true)]
	[string[]] $Args
)

$binDir = Join-Path $PSScriptRoot "llama-bin"
if (-not (Test-Path $binDir)) {
	Write-Error "llama-bin directory not found: $binDir"
	exit 1
}

# Prefer known llama executables, fall back to the first .exe found.
$preferred = @("llama-cli.exe", "llama.exe", "main.exe")
$exePath = $null
foreach ($name in $preferred) {
	$candidate = Join-Path $binDir $name
	if (Test-Path $candidate) {
		$exePath = $candidate
		break
	}
}

if (-not $exePath) {
	$firstExe = Get-ChildItem -Path $binDir -File | Where-Object { $_.Extension -eq ".exe" } | Select-Object -First 1
	if ($firstExe) {
		$exePath = $firstExe.FullName
	}
}

if (-not $exePath) {
	Write-Error "No llama executable found in $binDir"
	exit 1
}

& $exePath @Args
exit $LASTEXITCODE
