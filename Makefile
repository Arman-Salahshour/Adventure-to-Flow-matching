PYTHON ?= python
SRC = src

.PHONY: all regressor generate export validate clean

all: regressor generate export validate

# Three ResNet-18 seeds predicting the six rater scores.
regressor:
	$(PYTHON) $(SRC)/train_regressor.py

# Trains the generator and writes output/final_model.pth.
generate:
	$(PYTHON) $(SRC)/train_generator.py

# Writes output/images.pth and output/embeddings.pth.
export:
	$(PYTHON) export.py

validate:
	$(PYTHON) validate_project2.py

clean:
	rm -f output/*.pth
