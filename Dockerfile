# Build:  docker build -t fin-misinfo-detector .
# Run:    docker run -p 8000:8000 fin-misinfo-detector     -> http://localhost:8000/docs
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONPATH=/app/src
RUN python scripts/download_data.py && python -m fmd.train --quick
EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
