# Check-Services.ps1
# Quick service status checker for ACIS Trading Platform

Write-Host "`n╔════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║    ACIS TRADING PLATFORM SERVICES     ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════╝" -ForegroundColor Cyan

# Check if Docker is running
Write-Host "`n1️⃣ Docker Status:" -ForegroundColor Yellow
$dockerRunning = Get-Process docker -ErrorAction SilentlyContinue
if ($dockerRunning) {
    Write-Host "   ✅ Docker Desktop is running" -ForegroundColor Green

    # Check Docker daemon
    $dockerInfo = docker info 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✅ Docker daemon is responsive" -ForegroundColor Green
    } else {
        Write-Host "   ❌ Docker daemon not responding" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit
    }
} else {
    Write-Host "   ❌ Docker Desktop is not running" -ForegroundColor Red
    Write-Host "   Please start Docker Desktop first!" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit
}

# Check Docker Compose services
Write-Host "`n2️⃣ Docker Compose Services:" -ForegroundColor Yellow
$composeServices = docker-compose ps --services 2>$null
if ($LASTEXITCODE -eq 0) {
    $runningServices = docker-compose ps --services --filter "status=running" 2>$null
    $allServices = $composeServices -split "`n"

    foreach ($service in $allServices) {
        if ($service) {
            $isRunning = docker-compose ps -q $service 2>$null
            if ($isRunning) {
                Write-Host "   ✅ $service - Running" -ForegroundColor Green
            } else {
                Write-Host "   ⚠️  $service - Stopped" -ForegroundColor Yellow
            }
        }
    }
} else {
    Write-Host "   ⚠️  No docker-compose.yml found in current directory" -ForegroundColor Yellow
    Write-Host "   Current directory: $PWD" -ForegroundColor Gray
}

# Check port availability
Write-Host "`n3️⃣ Port Status:" -ForegroundColor Yellow
$portMapping = @{
    5432 = "PostgreSQL"
    6379 = "Redis"
    8000 = "API Server"
    80   = "Nginx"
    3000 = "Grafana"
    9090 = "Prometheus"
    3001 = "Frontend (if running)"
}

foreach ($port in $portMapping.Keys | Sort-Object) {
    $serviceName = $portMapping[$port]
    $connection = Test-NetConnection -ComputerName localhost -Port $port -WarningAction SilentlyContinue -InformationLevel Quiet

    if ($connection) {
        Write-Host "   ✅ Port $port ($serviceName) - Open" -ForegroundColor Green
    } else {
        Write-Host "   ⚠️  Port $port ($serviceName) - Closed" -ForegroundColor Gray
    }
}

# Check API endpoints
Write-Host "`n4️⃣ API Endpoints:" -ForegroundColor Yellow
$endpoints = @(
    @{Name="Health Check"; Url="http://localhost/health"; Method="GET"},
    @{Name="API Root"; Url="http://localhost/api"; Method="GET"},
    @{Name="Grafana UI"; Url="http://localhost:3000"; Method="GET"},
    @{Name="Prometheus UI"; Url="http://localhost:9090"; Method="GET"}
)

foreach ($endpoint in $endpoints) {
    try {
        $response = Invoke-WebRequest -Uri $endpoint.Url -Method $endpoint.Method -TimeoutSec 2 -ErrorAction Stop
        Write-Host "   ✅ $($endpoint.Name) - Responding (Status: $($response.StatusCode))" -ForegroundColor Green
    } catch {
        Write-Host "   ⚠️  $($endpoint.Name) - Not responding" -ForegroundColor Gray
    }
}

# Check container health
Write-Host "`n5️⃣ Container Health:" -ForegroundColor Yellow
$containers = docker ps --format "table {{.Names}}\t{{.Status}}" 2>$null
if ($containers) {
    $containerLines = $containers -split "`n" | Select-Object -Skip 1
    foreach ($line in $containerLines) {
        if ($line -match "healthy") {
            Write-Host "   ✅ $line" -ForegroundColor Green
        } elseif ($line -match "unhealthy") {
            Write-Host "   ❌ $line" -ForegroundColor Red
        } elseif ($line) {
            Write-Host "   ⚠️  $line" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "   No running containers found" -ForegroundColor Gray
}

# Check disk space (for Docker)
Write-Host "`n6️⃣ Docker Disk Usage:" -ForegroundColor Yellow
$dockerSystem = docker system df 2>$null
if ($LASTEXITCODE -eq 0) {
    docker system df
} else {
    Write-Host "   Unable to check disk usage" -ForegroundColor Gray
}

# Quick commands reference
Write-Host "`n📝 Quick Commands:" -ForegroundColor Cyan
Write-Host "   Start all services:  " -NoNewline; Write-Host "docker-compose up -d" -ForegroundColor Yellow
Write-Host "   Stop all services:   " -NoNewline; Write-Host "docker-compose down" -ForegroundColor Yellow
Write-Host "   View logs:           " -NoNewline; Write-Host "docker-compose logs -f [service-name]" -ForegroundColor Yellow
Write-Host "   Restart a service:   " -NoNewline; Write-Host "docker-compose restart [service-name]" -ForegroundColor Yellow
Write-Host "   Clean everything:    " -NoNewline; Write-Host "docker-compose down -v" -ForegroundColor Yellow

Write-Host "`n"
Read-Host "Press Enter to exit"
