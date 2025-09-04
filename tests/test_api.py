from pathlib import Path
import sys

# Assure l'import de l'app FastAPI depuis le projet
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from api import app


client = TestClient(app)


def test_health_ok():
	r = client.get("/health")
	assert r.status_code == 200
	data = r.json()
	assert data.get("status") == "ok"
	assert data.get("model_loaded") is True


def test_predict_empty_text_422():
	r = client.post("/predict", json={"text": ""})
	assert r.status_code == 422


def test_predict_ham_ok():
	r = client.post("/predict", json={"text": "Bonjour, réunion demain à 10h"})
	assert r.status_code == 200
	data = r.json()
	assert data["label"] in {"spam", "non-spam"}


def test_predict_spam_ok():
	r = client.post("/predict", json={"text": "Offre limitée!!! Cliquez ici pour gagner $$$"})
	assert r.status_code == 200
	data = r.json()
	assert data["label"] in {"spam", "non-spam"}
	assert "proba_spam" in data


def test_predict_long_text_ok():
	text = "lorem ipsum " * 500
	r = client.post("/predict", json={"text": text})
	assert r.status_code == 200
	data = r.json()
	assert data["label"] in {"spam", "non-spam"}


