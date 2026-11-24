FROM python:3.11-slim

RUN useradd -m -u 1000 nhsuser
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY outputs/ ./outputs/

RUN chown -R nhsuser:nhsuser /app
USER nhsuser

EXPOSE 8000
CMD ["uvicorn", "src.score_api:app", "--host", "0.0.0.0", "--port", "8000"]
