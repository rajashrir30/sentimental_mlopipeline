FROM python:3.11-slim AS build
WORKDIR /build
COPY serving/requirements.txt requirements.txt
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim AS runtime
WORKDIR /app
COPY --from=build /install /usr/local
COPY serving/app.py serving/app.py
COPY text_utils.py text_utils.py
COPY artifacts artifacts
EXPOSE 8000
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
