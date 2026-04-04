# Запуск HTTP API из каталога AIP (без uv).
# Из PowerShell в этой папке: .\run-api.ps1  (без .\ скрипт не найдётся — см. about_Command_Precedence)
# Или из любой папки: powershell -File "<полный_путь>\run-api.ps1"
# Двойной щелчок по .ps1 может быть заблокирован политикой выполнения — тогда используйте run-api.bat или команду выше.
Set-Location $PSScriptRoot
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python не найден в PATH."
    exit 1
}
Write-Host "Каталог: $(Get-Location)"
Write-Host "Сервер: http://127.0.0.1:8000/docs"
python -m uvicorn src.tumor_app.api:app --reload --host 127.0.0.1 --port 8000
