"""FastAPI inference service backed by a model in Amazon S3."""

from contextlib import asynccontextmanager
import os

import boto3
from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel, FiniteFloat

FEATURE_NAMES = [
    "age", "workclass", "education_num", "marital_status", "occupation",
    "relationship", "sex", "capital_gain", "capital_loss", "hours_per_week",
]
MODEL_KEY = "artifacts/current/model.joblib"


def download_model() -> str:
    """Use the EC2 instance role (or the local AWS profile) to download S3."""
    bucket = os.environ["ARTIFACT_BUCKET"]
    model_path = os.path.expanduser(
        os.environ.get("MODEL_PATH", "~/models/model.joblib")
    )
    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    temporary_path = model_path + ".download"
    boto3.client("s3").download_file(bucket, MODEL_KEY, temporary_path)
    os.replace(temporary_path, model_path)
    print(f"Downloaded model from s3://{bucket}/{MODEL_KEY}")
    return model_path


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail startup if the model cannot be downloaded or deserialized.
    app.state.model = joblib.load(download_model())
    yield


app = FastAPI(lifespan=lifespan)


class ScoreRequest(BaseModel):
    features: list[FiniteFloat]


@app.get("/healthz")
def healthz():
    if getattr(app.state, "model", None) is None:
        raise HTTPException(status_code=503, detail="Model is not ready")
    return {"status": "ok"}


@app.post("/score")
def score(req: ScoreRequest):
    if len(req.features) != len(FEATURE_NAMES):
        raise HTTPException(status_code=400, detail="Expected 10 features (adult income)")
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not ready")
    features = pd.DataFrame([req.features], columns=FEATURE_NAMES)
    prediction = int(model.predict(features)[0])
    return {
        "prediction": prediction,
        "label": "thu_nhap_cao" if prediction == 1 else "thu_nhap_thap",
    }


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
