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

	Tries pandas first; if it fails, split on the last comma for each row.
	"""
	try:
		df = pd.read_csv(csv_path)
		cols_lower = [c.lower() for c in df.columns]
		if "texte" not in cols_lower or "label" not in cols_lower:
			mapped = {}
			if len(df.columns) >= 1:
				mapped[df.columns[0]] = "texte"
			if len(df.columns) >= 2:
				mapped[df.columns[1]] = "label"
			df = df.rename(columns=mapped)
		return df
	except Exception:
		texts: list[str] = []
		labels: list[str] = []
		with open(csv_path, "r", encoding="utf-8") as f:
			_ = f.readline()  # header
			for raw in f:
				line = raw.rstrip("\n")
				if not line:
					continue
				try:
					text, label = line.rsplit(",", 1)
				except ValueError:
					continue
				texts.append(text.strip())
				labels.append(label.strip())
		return pd.DataFrame({"texte": texts, "label": labels})


def main() -> None:
	data_path = os.path.join("data", "emails.csv")
	if not os.path.exists(data_path):
		raise FileNotFoundError(f"Dataset not found at {data_path}")

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

	model: Pipeline = Pipeline(
		steps=[
			("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=20000)),
			("clf", LogisticRegression(max_iter=1000)),
		]
	)

	model.fit(X_train, y_train)

	y_pred = model.predict(X_test)
	acc = accuracy_score(y_test, y_pred)
	print(f"Accuracy: {acc:.4f}")
	print(classification_report(y_test, y_pred))

	out_path = "model.pkl"
	joblib.dump(model, out_path)
	print(f"Model saved to {out_path}")
	sys.exit(0)


if __name__ == "__main__":
	try:
		main()
	except Exception as exc:
		print(f"Training failed: {exc}", file=sys.stderr)
		sys.exit(1)



