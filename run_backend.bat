@echo off
echo Starting OLEA Secure API Gateway...
call venv\Scripts\activate.bat
python security_api.py
pause
