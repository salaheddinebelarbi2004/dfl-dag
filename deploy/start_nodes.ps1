# PowerShell Script to Start P2P Federated Learning Nodes
# Usage: powershell -ExecutionPolicy Bypass -File deploy/start_nodes.ps1

param(
    [int]$NumberOfNodes = 3,
    [int]$BasePort = 5000,
    [string]$Host = "127.0.0.1"
)

# Run from this script's directory so `p2p_node.py` resolves next to it.
Set-Location $PSScriptRoot

Write-Host "🚀 Starting P2P Federated Learning Nodes..." -ForegroundColor Cyan
Write-Host "   Nodes: $NumberOfNodes"
Write-Host "   Base Port: $BasePort"
Write-Host "   Host: $Host`n"

# Check if Python is installed
$pythonCheck = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Python not found. Install Python first." -ForegroundColor Red
    exit 1
}

Write-Host "✅ Python found: $pythonCheck" -ForegroundColor Green

# Start nodes
for ($i = 0; $i -lt $NumberOfNodes; $i++) {
    $port = $BasePort + $i
    $nodeName = "Node-$($i+1)"
    
    Write-Host "Starting $nodeName on $Host:$port..." -ForegroundColor Yellow
    
    # Start in new window
    Start-Process pwsh -ArgumentList "-NoExit", "-Command", "python p2p_node.py $Host $port" `
        -NoNewWindow
    
    # Wait between starts
    Start-Sleep -Seconds 1
}

Write-Host "`n✅ All nodes started!" -ForegroundColor Green
Write-Host "`nNext steps:`n"
Write-Host "1. Wait 5 seconds for nodes to initialize"
Write-Host "2. Connect nodes: 'connect $Host $BasePort' (from other node terminals)"
Write-Host "3. Send tasks: 'task sum [1,2,3]'"
Write-Host "4. Check results: 'result <task_id>'"
