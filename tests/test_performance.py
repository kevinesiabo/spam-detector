import os
from pathlib import Path
import joblib
import pandas as pd


def test_model_accuracy():
	root = Path(__file__).resolve().parents[1]
	assert (root / "model.pkl").exists(), "model.pkl manquant: exécuter train.py avant le test"
	df = pd.read_csv(root / "data" / "emails.csv")
	X = df["texte"].astype(str)
	y = df["label"].astype(str)
	model = joblib.load(root / "model.pkl")
	score = model.score(X, y)
	assert score >= 0.8, f"Accuracy trop faible : {score:.3f}"


