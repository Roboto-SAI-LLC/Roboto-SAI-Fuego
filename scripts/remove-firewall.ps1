#Requires -RunAsAdministrator

Write-Host "[INFO] Removing all Roboto SAI firewall rules..." -ForegroundColor Cyan

$rules = Get-NetFirewallRule -DisplayName "Roboto SAI*" -ErrorAction SilentlyContinue

if ($rules) {
    $count = ($rules | Measure-Object).Count
    $rules | Remove-NetFirewallRule
    Write-Host "[SUCCESS] Removed $count Roboto SAI firewall rule(s)" -ForegroundColor Green
} else {
    Write-Host "[INFO] No Roboto SAI firewall rules found" -ForegroundColor Yellow
}
