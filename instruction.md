# Запуск проекта

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt --no-deps
python -c "import torch; print(torch.cuda.is_available())"
python data_load.py
python main.py
```

# Повторный запуск

```powershell
.venv\Scripts\activate
python main.py
```

# Полное удаление проекта с зависимостями (включая CUDA-пакеты в venv)

```powershell
deactivate
Remove-Item -Recurse -Force .venv
Remove-Item -Recurse -Force artifacts
Remove-Item -Recurse -Force astifacts
```

