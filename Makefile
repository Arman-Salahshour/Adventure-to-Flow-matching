PYTHON ?= python
SRC = src

.PHONY: all deps regressor latents generator samples validate clean distclean

# Full pipeline. Every step skips work whose output already exists in output/.
all: deps regressor latents generator samples validate

# diffusers is the only package missing from the course venv.
# Installed into .deps, which is appended to the path last, so it never
# replaces anything the venv already provides.
deps:
	@$(PYTHON) -c "import sys; sys.path.append('.deps'); import diffusers" 2>/dev/null \
		|| $(PYTHON) -m pip install --quiet --target .deps diffusers

# Three ResNet-18 seeds predicting the six rater scores.
regressor: deps
	$(PYTHON) $(SRC)/train_regressor.py

# Encodes all cars with the frozen SD VAE into output/latents.pth.
latents: deps
	$(PYTHON) $(SRC)/encode_latents.py

# Trains the flow matching U-Net, writes output/fm_ema.pth.
generator: latents
	$(PYTHON) $(SRC)/train_fm.py

# Picks the 10 cars, writes images.pth, embeddings.pth and final_model.pth.
samples: regressor generator
	$(PYTHON) $(SRC)/select_images.py

# Loads the model the way the graders do and re-decodes the embeddings.
validate:
	$(PYTHON) validate_project2.py
	$(PYTHON) export.py

# Removes only the submission files, keeps every trained model.
clean:
	rm -f output/images.pth output/embeddings.pth output/final_model.pth

# Removes everything generated, including trained models.
distclean: clean
	rm -rf output/*.pth output/*.csv output/*.json output/*.png .deps
