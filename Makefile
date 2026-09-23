VENV = .venv-g07p2
PYTHON = $(VENV)/bin/python

AE_NF_SRC = src/ae_nf
VAE_SRC = src/vae
LATENT_FM_SRC = src/latent_fm

.PHONY: all setup ae_nf vae latent_fm clean

all: setup ae_nf

setup:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt

ae_nf: setup
	PYTHONPATH=$(AE_NF_SRC) $(PYTHON) $(AE_NF_SRC)/train/train_ae.py --latent_dim 128 --run_name final_model
	PYTHONPATH=$(AE_NF_SRC) $(PYTHON) $(AE_NF_SRC)/train/train_flow.py --ae_run_name final_model --run_name final_model
	cp output/final_model_flow_checkpoint.pth output/flow_checkpoint.pth
	cp output/final_model_checkpoint.pth output/final_model.pth
	PYTHONPATH=$(AE_NF_SRC) $(PYTHON) $(AE_NF_SRC)/generate.py --flow_run_name final_model

vae: setup
	PYTHONPATH=$(VAE_SRC) $(PYTHON) $(VAE_SRC)/train_vae.py

# Latent flow matching. Runs its own Makefile inside src/latent_fm, which
# writes only to output/latent_fm/. The venv path must be absolute there.
latent_fm: setup
	$(MAKE) -C $(LATENT_FM_SRC) PYTHON=$(CURDIR)/$(PYTHON)

clean:
	rm -rf $(VENV) output/final_model*.pth output/flow_checkpoint.pth output/images.pth output/embeddings.pth output/metrics
