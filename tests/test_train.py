import json

import joblib
import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score

from src.train import train

FEATURE_NAMES = [
    "age", "workclass", "education_num", "marital_status", "occupation",
    "relationship", "sex", "capital_gain", "capital_loss", "hours_per_week",
]
PARAMS = {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 2}


@pytest.fixture(autouse=True)
def isolated_training(tmp_path, monkeypatch):
    """Do not overwrite the real model, report, or local MLflow experiments."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_ID", raising=False)
    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    yield
    mlflow.end_run()
    mlflow.set_tracking_uri(previous_uri)


def _make_temp_data(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    X = rng.random((n, len(FEATURE_NAMES)))
    y = rng.integers(0, 2, size=n)
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["target"] = y
    train_path = str(tmp_path / "train.csv")
    eval_path = str(tmp_path / "holdout.csv")
    df.iloc[:160].to_csv(train_path, index=False)
    df.iloc[160:].to_csv(eval_path, index=False)
    return train_path, eval_path


def test_train_returns_float(tmp_path):
    train_path, eval_path = _make_temp_data(tmp_path)
    f1 = train(PARAMS, data_path=train_path, eval_path=eval_path)
    assert isinstance(f1, float)
    assert 0.0 <= f1 <= 1.0


def test_report_file_created(tmp_path):
    train_path, eval_path = _make_temp_data(tmp_path)
    f1 = train(PARAMS, data_path=train_path, eval_path=eval_path)
    report_path = tmp_path / "outputs/report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text())
    assert set(report) == {"f1_score", "accuracy"}
    assert report["f1_score"] == f1
    assert all(0.0 <= value <= 1.0 for value in report.values())


def test_model_file_created(tmp_path):
    train_path, eval_path = _make_temp_data(tmp_path)
    train(PARAMS, data_path=train_path, eval_path=eval_path)
    model_path = tmp_path / "models/model.joblib"
    assert model_path.is_file()
    model = joblib.load(model_path)
    holdout = pd.read_csv(eval_path)
    predictions = model.predict(holdout.drop(columns=["target"]))
    report = json.loads((tmp_path / "outputs/report.json").read_text())
    assert report["f1_score"] == f1_score(holdout.target, predictions)
    assert report["accuracy"] == accuracy_score(holdout.target, predictions)
