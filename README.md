Spam Detector  
Pipeline simple de détection de spam: TF-IDF + Logistic Regression.

Statut CI: ![CI](https://img.shields.io/badge/tests-passing-brightgreen)

Installation (Windows/PowerShell)
```powershell
cd D:\tp_school\spam-detector
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Entraînement
```powershell
python train.py
```

Tests
```powershell
pytest -q
```

Lancer l’application (Streamlit)
```powershell
python -m streamlit run streamlit_app.py --server.port 8520
```

API (FastAPI)
```powershell
python -m uvicorn api:app --app-dir . --reload --port 8000
```
Endpoints: `GET /health`, `POST /predict` (body {"text":"..."})

Makefile
```powershell
make train   # régénère model.pkl si data/train.py changent
make test    # entraîne puis lance pytest
```

CI GitHub Actions
Un workflow exécute l’installation, l’entraînement puis les tests à chaque push/PR.

Démo (ordre conseillé)
1. Notebook `notebooks/analysis.ipynb`: ROC, PR, CM, rapport de classification, seuil optimal
2. Streamlit: KPIs, styles de graphes (Seaborn/GGPlot/Sombre), normalisation CM, export CSV/PNG
3. API: `/health`, `/docs` puis `POST /predict`
4. Tests: `pytest` → tous verts

Données
`data/emails.csv` contient deux colonnes: `texte,label` (UTF-8, texte cité).

Exports (pour Figma)
`assets/roc.png`, `assets/pr.png`, `assets/cm.png`

Remontée de bug (résolu)
- MediaFileHandler (Streamlit) lors de relances → clear caches + relance port frais


