import os
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score


def load_dataset(csv_path: str) -> pd.DataFrame:
	"""Load dataset robustly even if 'texte' contains commas without quoting.

	Tries standard CSV parsing first; on failure, falls back to splitting on the last comma.
	"""
	try:
		df = pd.read_csv(csv_path)
		# Normalize expected column names
		cols_lower = [c.lower() for c in df.columns]
		if "texte" not in cols_lower or "label" not in cols_lower:
			# Try to coerce first two columns
			mapped = {}
			if len(df.columns) >= 1:
				mapped[df.columns[0]] = "texte"
			if len(df.columns) >= 2:
				mapped[df.columns[1]] = "label"
			df = df.rename(columns=mapped)
		return df
	except Exception:
		# Manual fallback: split by last comma
		texts: list[str] = []
		labels: list[str] = []
		with open(csv_path, "r", encoding="utf-8") as f:
			first_line = f.readline()
			for raw in f:
				line = raw.rstrip("\n")
				if not line:
					continue
				try:
					text, label = line.rsplit(",", 1)
				except ValueError:
					# Skip malformed line
					continue
				texts.append(text.strip())
				labels.append(label.strip())
		return pd.DataFrame({"texte": texts, "label": labels})


def main() -> None:
	data_path = os.path.join("data", "emails.csv")
	if not os.path.exists(data_path):
		raise FileNotFoundError(f"Dataset not found at {data_path}")

	# Load dataset (columns: 'texte', 'label')
	df = load_dataset(data_path)
	if "texte" not in df.columns or "label" not in df.columns:
		raise ValueError("CSV must contain columns 'texte' and 'label'")

	X = df["texte"].astype(str)
	y = df["label"].astype(str)

	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=0.2,
		random_state=42,
		stratify=y if len(y.unique()) > 1 else None,
	)

	# Build a simple baseline pipeline: TF-IDF + Logistic Regression
	model: Pipeline = Pipeline(
		steps=[
			("tfidf", TfidfVectorizer(stop_words=None, ngram_range=(1, 2), max_features=20000)),
			("clf", LogisticRegression(max_iter=1000, n_jobs=None)),
		]
	)

	model.fit(X_train, y_train)

	# Basic eval for sanity check
	y_pred = model.predict(X_test)
	acc = accuracy_score(y_test, y_pred)
	print(f"Accuracy: {acc:.4f}")
	print(classification_report(y_test, y_pred))

	# Save model
	out_path = "model.pkl"
	joblib.dump(model, out_path)
	print(f"Model saved to {out_path}")
	# Ensure clean success exit code in shells capturing native errors
	sys.exit(0)


if __name__ == "__main__":
	try:
		main()
	except Exception as exc:
		print(f"Training failed: {exc}", file=sys.stderr)
		sys.exit(1)

from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


DATA_PATH = Path("data/emails.csv")
MODEL_PATH = Path("model.pkl")


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset introuvable: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    # Harmoniser noms de colonnes attendus: 'texte' et 'label'
    if "texte" not in df.columns or "label" not in df.columns:
        raise ValueError("Le CSV doit contenir les colonnes 'texte' et 'label'.")
    df = df.dropna(subset=["texte", "label"])  # supprimer lignes vides
    X = df["texte"].astype(str)
    y = df["label"].astype(str)
    return X, y


def train_and_save(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline baseline: vectorisation sac-de-mots + régression logistique
    model = make_pipeline(
        CountVectorizer(),
        LogisticRegression(max_iter=1000)
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy (test): {acc:.4f}")

    joblib.dump(model, MODEL_PATH)
    print(f"Modèle sauvegardé → {MODEL_PATH}")


def main():
    try:
        X, y = load_data()
        train_and_save(X, y)
    except Exception as exc:
        print(f"Erreur: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

