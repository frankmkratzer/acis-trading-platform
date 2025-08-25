# Get-ProjectStructure.ps1
Write-Host "`n📁 ACIS Trading Platform Structure`n" -ForegroundColor Cyan

# Show src structure
Write-Host "Source Code Structure:" -ForegroundColor Yellow
Get-ChildItem -Path "src" -Recurse -File |
    Where-Object { $_.Extension -in ".py", ".js", ".ts", ".jsx", ".tsx", ".yml", ".yaml" } |
    ForEach-Object {
        $relativePath = $_.FullName.Replace($PWD.Path, "").TrimStart("\")
        Write-Host "  📄 $relativePath" -ForegroundColor Gray
    } | Select-Object -First 50

Write-Host "`nTotal files in src: " -NoNewline
(Get-ChildItem -Path "src" -Recurse -File | Measure-Object).Count

Write-Host "`nDocker services defined:" -ForegroundColor Yellow
$compose = Get-Content docker-compose.yml | ConvertFrom-Yaml -ErrorAction SilentlyContinue
if (-not $compose) {
    # Simple parsing if ConvertFrom-Yaml not available
    Select-String -Path docker-compose.yml -Pattern "^\s{2}\w+:" |
        ForEach-Object { Write-Host "  🐳 $($_.Line.Trim().TrimEnd(':'))" -ForegroundColor Cyan }
}