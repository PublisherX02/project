@echo off
echo Starting OLEA UI and Public Ngrok Tunnel...
call venv\Scripts\activate.bat
echo Launching Ngrok... (A new window will open)
start "Ngrok Tunnel" ngrok http 8501
echo Launching Streamlit...
streamlit run app.py
pause
