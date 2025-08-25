# fix-docker-path.ps1
Write-Host "🔧 Fixing Docker PATH Configuration" -ForegroundColor Cyan

# Check where docker.exe actually is
$dockerLocations = @(
    "C:\Program Files\Docker\Docker\resources\bin\docker.exe",
    "C:\ProgramData\DockerDesktop\version-bin\docker.exe",
    "$env:ProgramFiles\Docker\Docker\resources\bin\docker.exe"
)

$dockerFound = $null
foreach ($path in $dockerLocations) {
    if (Test-Path $path) {
        Write-Host "✅ Found docker.exe at: $path" -ForegroundColor Green
        $dockerFound = $path
        break
    }
}

if (-not $dockerFound) {
    Write-Host "❌ Could not find docker.exe" -ForegroundColor Red
    Write-Host "Searching for docker.exe..." -ForegroundColor Yellow
    $searchResult = Get-ChildItem -Path "C:\Program Files\Docker" -Filter "docker.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($searchResult) {
        $dockerFound = $searchResult.FullName
        Write-Host "✅ Found at: $dockerFound" -ForegroundColor Green
    }
}

if ($dockerFound) {
    # Get the directory
    $dockerDir = Split-Path $dockerFound -Parent

    # Add to current session PATH
    $env:PATH = "$dockerDir;$env:PATH"
    Write-Host "✅ Added to current session PATH: $dockerDir" -ForegroundColor Green

    # Test if it works now
    Write-Host "`nTesting docker command..." -ForegroundColor Yellow
    docker version > $null 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Docker command is now working!" -ForegroundColor Green

        # Test docker-compose
        docker-compose version > $null 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Docker-compose command is working!" -ForegroundColor Green
        } else {
            Write-Host "⚠️ Docker-compose not working, trying 'docker compose'..." -ForegroundColor Yellow
            docker compose version > $null 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Host "✅ 'docker compose' (new syntax) is working!" -ForegroundColor Green
            }
        }
    } else {
        Write-Host "❌ Docker still not responding" -ForegroundColor Red
    }

    # Offer to add to permanent PATH
    Write-Host "`n💡 To make this permanent, add to System PATH:" -ForegroundColor Yellow
    Write-Host "   $dockerDir" -ForegroundColor Cyan
    Write-Host "`nWould you like instructions to add this permanently? (Y/N)" -ForegroundColor Yellow
    $response = Read-Host

    if ($response -eq 'Y') {
        Write-Host "`nTo add permanently:" -ForegroundColor Cyan
        Write-Host "1. Press Windows Key + X" -ForegroundColor White
        Write-Host "2. Select 'System'" -ForegroundColor White
        Write-Host "3. Click 'Advanced system settings'" -ForegroundColor White
        Write-Host "4. Click 'Environment Variables'" -ForegroundColor White
        Write-Host "5. Under System Variables, select 'Path' and click 'Edit'" -ForegroundColor White
        Write-Host "6. Click 'New' and add: $dockerDir" -ForegroundColor White
        Write-Host "7. Click OK on all windows" -ForegroundColor White
        Write-Host "8. Restart PowerShell" -ForegroundColor White
    }
}

Write-Host "`nPress Enter to continue..."
Read-Host