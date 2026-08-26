Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Jupyter Book - Local Development" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Server starting at: http://localhost:8000" -ForegroundColor Yellow
Write-Host "Auto-reload: ON (changes refresh automatically)" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

sphinx-autobuild . _build/html --open-browser -a
