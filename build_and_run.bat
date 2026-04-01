@echo off
echo 🛡️  OLEA Insurance AI — Building Modern Containers...
echo.

:: 1. Stop existing containers to prevent port conflicts
docker-compose down

:: 2. Build and launch in detached mode
echo 🚀 Launching Secure API + Frontend Agent...
docker-compose up --build -d

echo.
echo ✅ Build Complete!
echo 🌐 Frontend: http://localhost:8502
echo 🔑 Admin Panel: http://localhost:8502/Admin_Logs (or port 8501 inside container)
echo.
echo 📜 To view real-time logs, run: docker-compose logs -f
pause
