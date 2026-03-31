# Первичная установка и запуск (все пакеты строго в .venv)

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --no-deps
.\.venv\Scripts\python.exe -m pip --version
.\.venv\Scripts\python.exe -c "import sys, torch; print(sys.executable); print(torch.cuda.is_available())"
.\.venv\Scripts\python.exe data_load.py
.\.venv\Scripts\python.exe main.py
```

# Повторный запуск проекта

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe main.py
```

# Полное удаление окружения и установленных зависимостей

```powershell
deactivate
Remove-Item -Recurse -Force .venv
Remove-Item -Recurse -Force artifacts
Remove-Item -Recurse -Force astifacts
```

