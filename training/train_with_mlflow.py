"""
train_with_mlflow.py  –  PT6 model + MLflow tracking & registry
----------------------------------------------------------------
Полностью совместим с PT6:
  • те же гиперпараметры (RandomForest, n_estimators=100, random_state=42)
  • тот же StandardScaler
  • сохраняет model.joblib в том же формате, что ожидает app.py
  • дополнительно: логирует всё в MLflow и регистрирует модель
"""

import os
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── MLflow config ─────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME     = "iris_randomforest"
MODEL_REGISTRY_NAME = "IrisClassifier"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# ── Hyperparameters (те же, что в PT6) ────────────────────────────────────────
PARAMS = {
    "n_estimators": 100,
    "random_state": 42,
    "test_size":    0.2,
}

# ── Data ──────────────────────────────────────────────────────────────────────
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=PARAMS["test_size"],
    random_state=PARAMS["random_state"],
    stratify=y,
)

# ── Preprocessing (тот же, что в PT6) ─────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── Train + MLflow run ────────────────────────────────────────────────────────
with mlflow.start_run(run_name="rf_iris_pt6") as run:
    print(f"MLflow Run ID : {run.info.run_id}")
    print(f"Experiment    : {EXPERIMENT_NAME}")

    # 1. Log parameters
    mlflow.log_params(PARAMS)

    # 2. Train
    model = RandomForestClassifier(
        n_estimators=PARAMS["n_estimators"],
        random_state=PARAMS["random_state"],
    )
    model.fit(X_train_scaled, y_train)

    # 3. Evaluate
    y_pred = model.predict(X_test_scaled)
    acc       = accuracy_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred, average="macro")
    precision = precision_score(y_test, y_pred, average="macro")
    recall    = recall_score(y_test, y_pred, average="macro")

    metrics = {
        "accuracy":  round(acc, 4),
        "f1_macro":  round(f1, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
    }
    mlflow.log_metrics(metrics)

    print(f"\nTest Accuracy : {acc * 100:.2f}%")
    print(f"F1 (macro)    : {f1:.4f}")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")

    # 4. Log classification report as artifact
    report = classification_report(y_test, y_pred, target_names=iris.target_names)
    print("\nClassification Report:\n", report)

    report_path = "/tmp/classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    mlflow.log_artifact(report_path, artifact_path="reports")

    # 5. Save model.joblib in PT6 format (model + scaler + metadata)
    model_artifact = {
        "model":         model,
        "scaler":        scaler,
        "feature_names": iris.feature_names,
        "target_names":  list(iris.target_names),
        "accuracy":      acc,
    }
    os.makedirs("/app/model", exist_ok=True)
    joblib_path = "/app/model/model.joblib"
    joblib.dump(model_artifact, joblib_path)
    mlflow.log_artifact(joblib_path, artifact_path="artifacts")

    # 6. Register sklearn model in MLflow Model Registry
    signature = mlflow.models.infer_signature(
        X_train_scaled,
        model.predict(X_train_scaled),
    )
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        registered_model_name=MODEL_REGISTRY_NAME,
    )

    print(f"\n✅ model.joblib saved → {joblib_path}")
    print(f"✅ Model registered   → '{MODEL_REGISTRY_NAME}' in MLflow Registry")
    print(f"✅ MLflow UI          → {MLFLOW_TRACKING_URI}")
