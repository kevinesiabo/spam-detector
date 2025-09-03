Spam Detector

Pipeline simple de détection de spam: TF-IDF + Logistic Regression.

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

Makefile
```powershell
make train   # régénère model.pkl si data/train.py changent
make test    # entraîne puis lance pytest
```

CI GitHub Actions
Un workflow exécute l’installation, l’entraînement puis les tests à chaque push/PR.

Données
`data/emails.csv` contient deux colonnes: `texte,label` (UTF-8, texte cité).


