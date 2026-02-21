@echo off
echo Starting OLEA AI Architecture (Docker) and Public Ngrok Tunnel...
echo Launching Ngrok... (A new window will open)
start "Ngrok Tunnel" ngrok http 8501
echo Launching Docker Containers...
docker-compose up
pause
