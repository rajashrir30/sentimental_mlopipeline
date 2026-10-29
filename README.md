# sentiment-mlops-pipeline

An end-to-end IMDB sentiment classification project with a reproducible preprocessing artifact, a scikit-learn baseline, a TensorFlow/Keras LSTM, MLflow tracking, and a FastAPI endpoint.

```mermaid
flowchart LR
  D[IMDB dataset] --> P[Clean and stratify]
  P --> T[Train Logistic Regression / LSTM]
  T --> M[MLflow runs and artifacts]
  M --> A[FastAPI serving]
  A --> U[React dashboard]
```

TF-IDF + Logistic Regression is the active serving model. The LSTM is trained
and evaluated for comparison, but is not used for live inference.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python src/preprocess.py --limit 5000
python src/train_baseline.py
python src/train_dl.py --epochs 2
uvicorn serving.app:app --reload
```

Copy `.env.example` to `.env` when you need to override local paths or MLflow
settings. The API exposes `/health` for diagnostics and `/ready` for a
deployment readiness check; neither endpoint exposes secrets or local paths.

The first preprocessing run downloads the public `imdb` dataset via Hugging Face. `--limit` limits each official split for quick local experiments; omit it for the full dataset. The resulting 80/10/10 stratified train/validation/test split is saved in `data/processed/imdb.json` and ignored by git.

OpenAPI docs are available at `http://localhost:8000/docs`. Example request:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"text\":\"A charming and brilliantly acted film.\"}"
```

## React dashboard

The polished dashboard lives in `frontend/` and includes a live sentiment analyzer, model performance comparison, and MLflow experiment history. Start the API first, then run:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The analyzer calls the FastAPI `POST /predict` endpoint. Performance and experiment views call the server-side `GET /mlflow/runs` endpoint, so MLflow tracking paths and credentials are never exposed to React. Run detail artifact links are served through the backend as well.

The API also exposes `GET /dataset/insights`, which reads the existing
`data/processed/imdb.json` artifact and returns measured split sizes, label
counts, duplicate counts, empty reviews, and review-length statistics. It does
not download data when the dashboard loads.

The dashboard expects the API process to run from the project root so the default `MLFLOW_TRACKING_URI=file:./mlruns` resolves correctly. You can override `MLFLOW_TRACKING_URI` and `MLFLOW_EXPERIMENT_NAME` for a remote tracking server.

## MLflow and containers

Training runs write to `./mlruns` by default. To use a server, set `MLFLOW_TRACKING_URI=http://localhost:5000` before training. Start the tracking server and API with `docker compose up --build`; train the model first so `artifacts/baseline.joblib` exists for the serving image.

The Compose serving container mounts the local `mlruns/` directory read-only so
the dashboard can display the same recorded runs as the MLflow service.

The training image runs `src/train_baseline.py` by default:

```bash
docker build -f training/Dockerfile -t sentiment-training .
docker run --rm -v "${PWD}/data:/app/data" -v "${PWD}/artifacts:/app/artifacts" sentiment-training
```

## Design notes

- `src/preprocess.py` is the single source of cleaned text and split assignments, so both models use identical data.
- The baseline model is the default serving artifact because it is small and has no TensorFlow runtime dependency at inference.
- Production deployments should pin dependencies, use a managed artifact store, add authentication, and register/promote models in MLflow.
- Local `mlruns/` and generated dataset files are not guaranteed to persist on cloud redeployments unless the hosting platform provides persistent storage.

## Verification

Run the regression checks with `pytest -q` after installing the project
requirements. The tests cover text normalization, API readiness, and a real
prediction from the saved baseline artifact. Build the dashboard with
`cd frontend && npm run build`.

