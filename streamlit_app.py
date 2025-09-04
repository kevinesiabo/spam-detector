import os
import io
import sys
import subprocess
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Nettoyage des caches pour éviter les erreurs de médias/caches obsolètes
try:
	st.cache_data.clear()
	st.cache_resource.clear()
except Exception:
	pass

st.set_page_config(page_title="Spam Detector", layout="centered")

st.title("Spam Detector – Analyse minimaliste")
st.caption("Design minimal, images légères, strict à l'essentiel")

# Style des graphes
style_label_to_mpl = {
	"Par défaut": "default",
	"Seaborn": "seaborn-v0_8-whitegrid",
	"GGPlot": "ggplot",
	"Sombre": "dark_background",
}
style_choice = st.selectbox("Style des graphes", list(style_label_to_mpl.keys()), index=1)
plt.style.use(style_label_to_mpl[style_choice])

@st.cache_resource
def load_model():
	root = os.path.dirname(__file__)
	model_path = os.path.join(root, "model.pkl")
	return joblib.load(model_path)

@st.cache_data
def load_data(mod_time: float):
	root = os.path.dirname(__file__)
	path = os.path.join(root, "data", "emails.csv")
	return pd.read_csv(path)

st.title("Spam Detector – Analyse minimaliste")
st.caption("Design minimal, images légères, strict à l'essentiel")

# Gestion modèle manquant
model = None
try:
	model = load_model()
except Exception as e:
	st.error("model.pkl introuvable. Entraînez le modèle pour continuer.")
	if st.button("Entraîner maintenant"):
		with st.spinner("Entraînement en cours..."):
			root = os.path.dirname(__file__)
			python_exec = sys.executable
			res = subprocess.run([python_exec, os.path.join(root, "train.py")], capture_output=True, text=True, cwd=root)
			st.text(res.stdout)
			if res.returncode == 0:
				st.success("Modèle entraîné. Rechargez la page.")
			else:
				st.error("Échec de l'entraînement. Voir logs ci-dessus.")
	st.stop()

# Chargement données avec invalidation sur mtime
csv_path = os.path.join(os.path.dirname(__file__), "data", "emails.csv")
mod_time = os.path.getmtime(csv_path) if os.path.exists(csv_path) else 0.0
df = load_data(mod_time)

X = df["texte"].astype(str)
y = df["label"].astype(str)

# Probabilités globales pour optimisation de seuil
if hasattr(model, "predict_proba"):
	y_proba = model.predict_proba(X)[:, 1]
else:
	y_proba = None

# Seuil optimal (F1) si proba dispo, sinon 0.5
default_threshold = 0.5
if y_proba is not None:
	from sklearn.metrics import precision_recall_curve
	y_bin = (y == "spam").astype(int)
	prec, rec, thr = precision_recall_curve(y_bin, y_proba)
	# thr length = len(prec)-1
	import numpy as np
	f1 = (2 * prec[:-1] * rec[:-1]) / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
	best_idx = int(np.nanargmax(f1))
	default_threshold = float(thr[best_idx])

col1, col2 = st.columns(2)
with col1:
	threshold = st.slider("Seuil de classification", 0.0, 1.0, default_threshold, 0.01)
with col2:
	st.write("Nombre d'emails:", len(df))

# Prédictions au seuil choisi
if y_proba is not None:
	y_pred = (y_proba >= threshold).astype(int)
	labels_pred = pd.Series(np.where(y_pred == 1, "spam", "non-spam"))
else:
	labels_pred = model.predict(X)

# Métriques clés
acc = accuracy_score(y, labels_pred)
prec = precision_score(y, labels_pred, pos_label="spam")
rec = recall_score(y, labels_pred, pos_label="spam")
f1 = f1_score(y, labels_pred, pos_label="spam")

met1, met2, met3, met4 = st.columns(4)
met1.metric("Accuracy", f"{acc:.3f}")
met2.metric("Précision (spam)", f"{prec:.3f}")
met3.metric("Rappel (spam)", f"{rec:.3f}")
met4.metric("F1 (spam)", f"{f1:.3f}")

# Boutons utilitaires sur le seuil
btn1, btn2 = st.columns(2)
with btn1:
	if st.button("Définir seuil optimal (F1)"):
		st.session_state["threshold"] = default_threshold
with btn2:
	if st.button("Réinitialiser (0.5)"):
		st.session_state["threshold"] = 0.5


st.subheader("Courbe ROC & PR")
if y_proba is not None:
	y_bin = (y == "spam").astype(int)
	fpr, tpr, _ = roc_curve(y_bin, y_proba)
	roc_auc = auc(fpr, tpr)
	fig_roc, ax = plt.subplots(figsize=(4,3))
	ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
	ax.plot([0,1],[0,1],"k--", linewidth=0.8)
	ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC"); ax.legend()
	st.pyplot(fig_roc, clear_figure=True)

	precision, recall, _ = precision_recall_curve(y_bin, y_proba)
	fig_pr, ax = plt.subplots(figsize=(4,3))
	ax.plot(recall, precision)
	ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("PR")
	st.pyplot(fig_pr, clear_figure=True)
else:
	st.info("Probabilités non disponibles pour ce modèle.")

st.subheader("Matrice de confusion")
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
norm_label_to_arg = {
	"Aucune": None,
	"Vraie (par ligne)": "true",
	"Prédite (par colonne)": "pred",
}
norm_choice = st.selectbox("Normalisation", list(norm_label_to_arg.keys()), index=0)
norm_arg = norm_label_to_arg[norm_choice]
cm = confusion_matrix(y, labels_pred, labels=["non-spam", "spam"], normalize=norm_arg)
fig_cm, ax = plt.subplots(figsize=(4,3))
fmt = ".2f" if norm_arg is not None else "d"
sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", xticklabels=["non-spam","spam"], yticklabels=["non-spam","spam"], ax=ax)
ax.set_xlabel("Prédit"); ax.set_ylabel("Réel")
st.pyplot(fig_cm, clear_figure=True)

# Téléchargements: métriques CSV + figures PNG
st.subheader("Export")
import csv
buf = io.StringIO()
writer = csv.writer(buf)
writer.writerow(["metric","value"])
writer.writerow(["accuracy", f"{acc:.4f}"])
writer.writerow(["precision_spam", f"{prec:.4f}"])
writer.writerow(["recall_spam", f"{rec:.4f}"])
writer.writerow(["f1_spam", f"{f1:.4f}"])
metrics_csv = buf.getvalue().encode("utf-8")

col_dl1, col_dl2, col_dl3 = st.columns(3)
with col_dl1:
    st.download_button("Télécharger métriques (CSV)", metrics_csv, file_name="metrics.csv", mime="text/csv")
with col_dl2:
    png_roc = io.BytesIO()
    if y_proba is not None:
        fig_roc.savefig(png_roc, format="png", dpi=120, bbox_inches="tight")
        st.download_button("Télécharger ROC (PNG)", png_roc.getvalue(), file_name="roc.png", mime="image/png")
with col_dl3:
    png_cm = io.BytesIO()
    fig_cm.savefig(png_cm, format="png", dpi=120, bbox_inches="tight")
    st.download_button("Télécharger CM (PNG)", png_cm.getvalue(), file_name="confusion_matrix.png", mime="image/png")

st.subheader("Démo de prédiction")
text = st.text_area("Saisir un email (texte)", "Merci pour votre retour concernant le projet à valider")
if st.button("Prédire"):
	pred = model.predict([text])[0]
	st.success(f"Classe: {pred}")
	if y_proba is not None:
		proba = model.predict_proba([text])[0,1]
		st.write(f"Probabilité spam: {proba:.3f}")


