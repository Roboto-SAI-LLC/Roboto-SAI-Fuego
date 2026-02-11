#Requires -RunAsAdministrator

<#
.SYNOPSIS
  Simple, reliable firewall setup for Roboto SAI - allows only 127.0.0.1 loopback traffic
#>

$ErrorActionPreference = 'Stop'

Write-Host "[INFO] Starting Roboto SAI firewall setup (simple mode)..." -ForegroundColor Cyan

# Remove any existing Roboto SAI rules
Write-Host "[INFO] Removing existing Roboto SAI firewall rules..."
Get-NetFirewallRule -DisplayName "Roboto SAI - *" -ErrorAction SilentlyContinue | Remove-NetFirewallRule -ErrorAction SilentlyContinue

# Find executables
$llamaServer = "R:\Repos\Roboto-SAI-2026\apps\model-training\tools\llama-bin\llama-server.exe"
$node = (Get-Command node -ErrorAction SilentlyContinue).Source
$code = (Get-Command code -ErrorAction SilentlyContinue).Source
$codeInsiders = (Get-Command code-insiders -ErrorAction SilentlyContinue).Source

# Verify critical programs exist
if (-not (Test-Path $llamaServer)) {
    Write-Host "[ERROR] llama-server.exe not found at: $llamaServer" -ForegroundColor Red
    exit 1
}

Write-Host "[INFO] Found llama-server: $llamaServer" -ForegroundColor Green

# Create simple rules for llama-server (most critical)
Write-Host "[INFO] Creating firewall rules for llama-server.exe..."

New-NetFirewallRule `
    -DisplayName "Roboto SAI - Allow llama-server loopback" `
    -Direction Inbound `
    -Action Allow `
    -Program $llamaServer `
    -LocalAddress 127.0.0.1 `
    -RemoteAddress 127.0.0.1 `
    -Protocol TCP `
    -Enabled True | Out-Null

New-NetFirewallRule `
    -DisplayName "Roboto SAI - Block llama-server external" `
    -Direction Inbound `
    -Action Block `
    -Program $llamaServer `
    -LocalAddress Any `
    -RemoteAddress Any `
    -Protocol TCP `
    -Enabled True | Out-Null

Write-Host "[INFO] Created 2 firewall rules for llama-server" -ForegroundColor Green

# Create rules for Node.js if found
if ($node) {
    Write-Host "[INFO] Creating firewall rules for node.exe..."
    
    New-NetFirewallRule `
        -DisplayName "Roboto SAI - Allow node loopback" `
        -Direction Inbound `
        -Action Allow `
        -Program $node `
        -LocalAddress 127.0.0.1 `
        -RemoteAddress 127.0.0.1 `
        -Protocol TCP `
        -Enabled True | Out-Null
    
    New-NetFirewallRule `
        -DisplayName "Roboto SAI - Block node external" `
        -Direction Inbound `
        -Action Block `
        -Program $node `
        -LocalAddress Any `
        -RemoteAddress Any `
        -Protocol TCP `
        -Enabled True | Out-Null
    
    Write-Host "[INFO] Created 2 firewall rules for node.exe" -ForegroundColor Green
}

# Summary
Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Roboto SAI Firewall Setup Complete" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Rules created:" -ForegroundColor Yellow
Get-NetFirewallRule -DisplayName "Roboto SAI - *" | Format-Table DisplayName, Direction, Action, Enabled -AutoSize
Write-Host ""
Write-Host "[SUCCESS] llama-server can only accept connections from 127.0.0.1" -ForegroundColor Green
Write-Host "[INFO] To remove these rules, run: Get-NetFirewallRule -DisplayName 'Roboto SAI - *' | Remove-NetFirewallRule" -ForegroundColor Gray
Write-Host ""
