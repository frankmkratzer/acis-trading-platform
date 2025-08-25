# Test-TradingPlatform.ps1
Write-Host "`n🚀 ACIS Trading Platform Test Suite`n" -ForegroundColor Cyan

function Test-AllServices {
    $results = @()

    # Test Docker
    Write-Host "Checking Docker..." -ForegroundColor Yellow
    $dockerRunning = docker info 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Docker is running" -ForegroundColor Green
        $results += $true
    } else {
        Write-Host "❌ Docker is not running - please start Docker Desktop" -ForegroundColor Red
        return
    }

    # Test Docker Compose Services
    Write-Host "`nChecking Docker Compose services..." -ForegroundColor Yellow
    docker-compose ps

    # Test specific services
    $services = @("postgres", "redis", "trading-engine", "api", "nginx")
    foreach ($service in $services) {
        $status = docker-compose ps -q $service 2>$null
        if ($status) {
            Write-Host "✅ $service is running" -ForegroundColor Green
            $results += $true
        } else {
            Write-Host "⚠️ $service is not running" -ForegroundColor Yellow
            $results += $false
        }
    }

    # Test API endpoint
    Write-Host "`nTesting API endpoints..." -ForegroundColor Yellow
    try {
        $response = Invoke-WebRequest -Uri "http://localhost/health" -ErrorAction Stop
        Write-Host "✅ API is responding" -ForegroundColor Green
        $results += $true
    } catch {
        Write-Host "⚠️ API not responding yet" -ForegroundColor Yellow
        $results += $false
    }

    # Summary
    $passed = ($results | Where-Object { $_ -eq $true }).Count
    $total = $results.Count
    Write-Host "`n📊 Results: $passed/$total checks passed" -ForegroundColor Cyan
}

# Run the tests
Test-AllServices