$ErrorActionPreference = 'Stop'
python -m venv backend/.venv
backend\.venv\Scripts\python.exe -m pip install --upgrade pip
backend\.venv\Scripts\pip.exe install -r backend\requirements.txt
npm --prefix frontend install
