"""Train and log a compact Keras LSTM model."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import classification_metrics
from mlflow_utils import configure_mlflow, log_metrics_and_artifact
from preprocess import DEFAULT_OUTPUT, load_processed


def train(data_path=DEFAULT_OUTPUT, model_path="artifacts/lstm.keras", epochs=3):
    started_at = time.perf_counter()
    data = load_processed(data_path)
    tokenizer = Tokenizer(num_words=20_000, oov_token="<OOV>")
    tokenizer.fit_on_texts(data["train"]["text"])
    encode = lambda split: pad_sequences(tokenizer.texts_to_sequences(data[split]["text"]), maxlen=250, padding="post", truncating="post")
    x_train, x_val, x_test = (encode(s) for s in ("train", "validation", "test"))
    y_train, y_val, y_test = (np.asarray(data[s]["label"]) for s in ("train", "validation", "test"))
    tf.random.set_seed(42)
    model = Sequential([Embedding(20_000, 64, mask_zero=True), LSTM(64), Dense(1, activation="sigmoid")])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(
        x_train, y_train, validation_data=(x_val, y_val), epochs=epochs,
        batch_size=128, verbose=2,
        callbacks=[EarlyStopping(monitor="val_loss", patience=1, restore_best_weights=True)],
    )
    val_pred = (model.predict(x_val, verbose=0).ravel() >= 0.5).astype(int)
    test_pred = (model.predict(x_test, verbose=0).ravel() >= 0.5).astype(int)
    validation_metrics = classification_metrics(y_val, val_pred)
    metrics = classification_metrics(y_test, test_pred)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    tokenizer_path = Path(model_path).with_suffix(".tokenizer.json")
    tokenizer_path.write_text(tokenizer.to_json(), encoding="utf-8")
    configure_mlflow()
    with mlflow.start_run(run_name="keras-lstm"):
        log_metrics_and_artifact(
            {**{f"validation_{k}": v for k, v in validation_metrics.items()},
             **{f"test_{k}": v for k, v in metrics.items()}},
            {"model": "lstm", "epochs": epochs, "max_len": 250, "seed": 42,
             "train_samples": len(y_train), "validation_samples": len(y_val),
             "test_samples": len(y_test),
             "training_seconds": round(time.perf_counter() - started_at, 3)}, model_path,
        )
        mlflow.log_artifact(str(tokenizer_path))
    result = {"validation": validation_metrics, "test": metrics}
    print(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model", default="artifacts/lstm.keras")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    train(args.data, args.model, args.epochs)
