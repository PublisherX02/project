FROM python:3.10-slim
WORKDIR /app
COPY requirements-app.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements-app.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
