import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix


def main() -> None:
	root = os.path.dirname(os.path.dirname(__file__))
	data_path = os.path.join(root, "data", "emails.csv")
	model_path = os.path.join(root, "model.pkl")
	assets_dir = os.path.join(root, "assets")
	os.makedirs(assets_dir, exist_ok=True)

	model = joblib.load(model_path)
	df = pd.read_csv(data_path)
	X = df["texte"].astype(str)
	y = df["label"].astype(str)

	y_bin = (y == "spam").astype(int)
	if hasattr(model, "predict_proba"):
		proba = model.predict_proba(X)[:, 1]
	else:
		# fallback via decision_function -> map to [0,1] using sigmoid-ish scaling
		if hasattr(model, "decision_function"):
			df_scores = model.decision_function(X)
			# simple min-max to [0,1]
			m, M = float(np.min(df_scores)), float(np.max(df_scores))
			proba = (df_scores - m) / (M - m + 1e-9)
		else:
			proba = (model.predict(X) == "spam").astype(float)

	# ROC
	fpr, tpr, _ = roc_curve(y_bin, proba)
	roc_auc = auc(fpr, tpr)
	fig, ax = plt.subplots(figsize=(6, 4))
	ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
	ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
	ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC"); ax.legend()
	fig.savefig(os.path.join(assets_dir, "roc.png"), dpi=120, bbox_inches="tight")
	plt.close(fig)

	# PR
	precision, recall, _ = precision_recall_curve(y_bin, proba)
	fig, ax = plt.subplots(figsize=(6, 4))
	ax.plot(recall, precision)
	ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-Recall")
	fig.savefig(os.path.join(assets_dir, "pr.png"), dpi=120, bbox_inches="tight")
	plt.close(fig)

	# CM
	threshold = 0.5
	pred = (proba >= threshold).astype(int)
	labels_pred = np.where(pred == 1, "spam", "non-spam")
	cm = confusion_matrix(y, labels_pred, labels=["non-spam", "spam"])
	fig, ax = plt.subplots(figsize=(6, 4))
	im = ax.imshow(cm, cmap="Blues")
	for (i, j), v in np.ndenumerate(cm):
		ax.text(j, i, str(v), ha='center', va='center')
	ax.set_xticks([0,1]); ax.set_xticklabels(["non-spam","spam"]) 
	ax.set_yticks([0,1]); ax.set_yticklabels(["non-spam","spam"]) 
	ax.set_xlabel("Prédit"); ax.set_ylabel("Réel")
	fig.colorbar(im, ax=ax)
	fig.savefig(os.path.join(assets_dir, "cm.png"), dpi=120, bbox_inches="tight")
	plt.close(fig)

	print("Figures exportées dans:", assets_dir)


if __name__ == "__main__":
	main()


