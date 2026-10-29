FROM node:22-alpine AS frontend-build
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend ./
ENV VITE_API_URL=
RUN npm run build

FROM python:3.11-slim AS python-deps
WORKDIR /build
COPY serving/requirements.txt requirements.txt
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim AS runtime
WORKDIR /app
COPY --from=python-deps /install /usr/local
COPY serving/app.py serving/app.py
COPY text_utils.py text_utils.py
COPY artifacts artifacts
COPY --from=frontend-build /frontend/dist frontend/dist
ENV MODEL_PATH=artifacts/baseline.joblib
ENV MLFLOW_TRACKING_URI=file:./mlruns
RUN addgroup --system app && adduser --system --ingroup app app \
    && chown -R app:app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/ready')"
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
