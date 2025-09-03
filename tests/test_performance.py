import os
import joblib
import pandas as pd


def test_model_accuracy():
	assert os.path.exists("model.pkl"), "model.pkl manquant: exécuter train.py avant le test"
	df = pd.read_csv("data/emails.csv")
	X = df["texte"].astype(str)
	y = df["label"].astype(str)
	model = joblib.load("model.pkl")
	score = model.score(X, y)
	assert score >= 0.8, f"Accuracy trop faible : {score:.3f}"


