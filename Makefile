PYTHON=python

# Rebuild model whenever training script or dataset changes
train: model.pkl

model.pkl: train.py data/emails.csv
	$(PYTHON) train.py

.PHONY: train clean test

clean:
	-$(RM) model.pkl

test: train
	pytest -q

PYTHON := python

# Règle principale: réentraîner si les sources changent
model.pkl: train.py data/emails.csv
	$(PYTHON) train.py

.PHONY: train clean

train: model.pkl

clean:
	-$(RM) model.pkl

