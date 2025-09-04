from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

app = FastAPI(title="Spam Detector API", version="1.0.0")

try:
	root = os.path.dirname(__file__)
	model_path = os.path.join(root, "model.pkl")
	model = joblib.load(model_path)
except Exception as exc:
	model = None
	# API démarre quand même; endpoint renverra 503 si modèle manquant


class PredictRequest(BaseModel):
	text: str


class PredictResponse(BaseModel):
	label: str
	proba_spam: float | None = None


@app.get("/health")
def health() -> dict:
	return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
	if model is None:
		raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

	text = (req.text or "").strip()
	if not text:
		raise HTTPException(status_code=422, detail="Field 'text' must be non-empty")

	label = model.predict([text])[0]
	proba = None
	if hasattr(model, "predict_proba"):
		proba = float(model.predict_proba([text])[0, 1])
	return PredictResponse(label=label, proba_spam=proba)


