FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements-app.txt ./
RUN pip install --no-cache-dir -r requirements-app.txt && useradd --create-home --uid 10001 sentinel
COPY dashboard ./dashboard
COPY api ./api
COPY .streamlit ./.streamlit
USER sentinel
EXPOSE 8000 8501
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
