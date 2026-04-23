import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --------------------------------------------------
# App initialization
# --------------------------------------------------
app = FastAPI(
    title="Iris ML API",
    description="Serves predictions from a trained RandomForest model on the Iris dataset.",
    version="1.0.0",
)

# --------------------------------------------------
# Load model artifact once at startup
# --------------------------------------------------
MODEL_PATH = "/app/model/model.joblib"
try:
    artifact = joblib.load(MODEL_PATH)
    model         = artifact["model"]
    scaler        = artifact["scaler"]
    feature_names = artifact["feature_names"]
    target_names  = artifact["target_names"]
    model_accuracy = artifact["accuracy"]
except FileNotFoundError:
    raise RuntimeError(f"Model file '{MODEL_PATH}' not found. Run train_model.py first.")

# --------------------------------------------------
# Request / Response schemas
# --------------------------------------------------
class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, example=5.1, description="Sepal length in cm")
    sepal_width:  float = Field(..., gt=0, example=3.5, description="Sepal width in cm")
    petal_length: float = Field(..., gt=0, example=1.4, description="Petal length in cm")
    petal_width:  float = Field(..., gt=0, example=0.2, description="Petal width in cm")

class PredictionResponse(BaseModel):
    predicted_class_id:   int
    predicted_class_name: str
    probabilities:        dict[str, float]

# --------------------------------------------------
# Endpoints
# --------------------------------------------------
@app.get("/", summary="Health check")
def root():
    """Returns a simple message confirming the API is running."""
    return {"message": "ML API is running"}

@app.post("/predict", response_model=PredictionResponse, summary="Predict Iris species")
def predict(features: IrisFeatures):
    """
    Accepts four Iris measurements, runs them through the trained
    RandomForest model and returns the predicted species.
    """
    try:
        X = np.array([[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width,
        ]])
        X_scaled   = scaler.transform(X)
        class_id   = int(model.predict(X_scaled)[0])
        class_name = target_names[class_id]
        proba      = model.predict_proba(X_scaled)[0]
        proba_dict = {name: round(float(p), 4) for name, p in zip(target_names, proba)}

        return PredictionResponse(
            predicted_class_id=class_id,
            predicted_class_name=class_name,
            probabilities=proba_dict,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/model-info", summary="Model metadata")
def model_info():
    """Returns metadata about the loaded model."""
    return {
        "model_type": type(model).__name__,
        "features":   list(feature_names),
        "classes":    list(target_names),
        "test_accuracy": round(model_accuracy, 4),
        "model_file": MODEL_PATH,
    }