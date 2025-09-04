from pathlib import Path
import subprocess
import sys
import os
import shutil


def pytest_sessionstart(session):
	root = Path(__file__).resolve().parents[1]
	# Assurer que tous les chemins relatifs pointent vers la racine du projet
	os.chdir(root)
	model = root / "model.pkl"
	data = root / "data" / "emails.csv"
	if not data.exists():
		raise SystemExit("Dataset manquant: data/emails.csv")
	if not model.exists():
		res = subprocess.run([sys.executable, str(root / "train.py")], cwd=str(root))
		if res.returncode != 0:
			raise SystemExit("exécuter train.py avant le test")

	# Garantir la présence de model.pkl là où PyTest est lancé
	start_cwd = Path.cwd()
	try:
		if start_cwd != root:
			shutil.copy2(model, start_cwd / "model.pkl")
	except Exception:
		pass


