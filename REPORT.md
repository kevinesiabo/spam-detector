# Rendu – Spam Detector

## 1) Objectif
Détection de spam par régression logistique avec vectorisation TF‑IDF, analyse et démonstration via Streamlit et FastAPI.

## 2) Données
`data/emails.csv` (colonnes `texte,label`). Pré-traitement simple; modèle binaire `spam` / `non-spam`.

## 3) Modèle
- Pipeline: `TfidfVectorizer(ngram_range=(1,2), max_features=20000)` + `LogisticRegression(max_iter=1000)`
- Entraînement: `python train.py`
- Export modèle: `model.pkl`

## 4) Résultats principaux
- Accuracy (validation interne): ≥ 0.8 (voir logs d’entraînement)
- Graphes: ROC (AUC), Precision‑Recall, Matrice de confusion
- Exports PNG: `assets/roc.png`, `assets/pr.png`, `assets/cm.png`

## 5) Notebook
`notebooks/analysis.ipynb`:
- Chargement des données et du modèle
- Rapport de classification
- Courbes ROC/PR, CM
- Importance des caractéristiques (TF‑IDF + coefficients)
- Seuil optimal (F1)

## 6) Dashboard (Streamlit)
Lancement: `python -m streamlit run streamlit_app.py --server.port 8520`

Fonctionnalités:
- KPIs (accuracy, précision, rappel, F1)
- Seuil ajustable + bouton « Seuil optimal (F1) »
- Styles de graphes (Seaborn, GGPlot, Sombre)
- Normalisation de la CM (aucune, vraie, prédite)
- Export CSV/PNG (métriques et figures)
- Conception minimaliste/éco (caching, figures légères)

## 7) API (FastAPI)
Lancement: `python -m uvicorn api:app --app-dir . --reload --port 8000`
- `GET /health` → statut et chargement du modèle
- `POST /predict` → body `{ "text": "..." }` → `{ "label": "spam|non-spam", "proba_spam": float }`

Exemples:
```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"text\":\"Bonjour\"}"
```

## 8) Tests et CI
Tests: `pytest -q` (7 passed). Un hook entraîne automatiquement si `model.pkl` est absent.
CI GitHub Actions: installation → entraînement → tests à chaque push/PR.

## 9) Bug report (résolu)
Problème: MediaFileHandler (caches Streamlit) au rechargement.
Correctifs: purge caches au démarrage, relance sur port frais.

## 10) Guide de démonstration (5–7 min)
1. Notebook: ROC/PR/CM, seuil optimal
2. Dashboard: KPIs, styles, normalisation, exports
3. API: `/health`, `/docs`, `POST /predict`
4. Tests: `pytest` → tous verts


