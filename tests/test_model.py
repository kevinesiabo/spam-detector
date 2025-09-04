import os
from pathlib import Path


def test_model_exists():
	root = Path(__file__).resolve().parents[1]
	assert (root / "model.pkl").exists()


