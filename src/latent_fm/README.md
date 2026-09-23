# Latent flow matching

Generates cars for Moira in the latent space of a frozen, pretrained image
autoencoder.

1. **Encode.** The Stable Diffusion VAE (`stabilityai/sd-vae-ft-mse`, frozen)
   compresses each 3x96x128 car into a 4x12x16 latent.
2. **Generate.** A conditional U-Net (32.8M parameters) is trained with flow
   matching to turn noise into latents, conditioned on the six scores through
   FiLM, with classifier-free guidance. Early stopping on held-out loss.
3. **Select.** 512 candidates are sampled at Moira = 9 and scored by a
   fine-tuned ResNet-18. Near-copies of real cars are dropped and the ten best,
   mutually different candidates are kept.

Each embedding is 774 values: 768 for the starting noise, 6 for the scores.
Decoding is deterministic, so the embeddings reproduce the images exactly.

## Run

From the repo root:

    make latent_fm

or in the background:

    mkdir -p logs && sbatch src/latent_fm/run.slurm

Every step skips work already done. All files go to `output/latent_fm/`,
separate from the main deliverable in `output/`. The working notebook is
`notebooks/latent_fm.ipynb`; the state of the generator is in `EVALUATION.md`.

## Files

- `constants.py` paths, seed, rater names
- `data.py`, `eda.py`, `viz.py` loading, preference analysis, plots
- `regressor.py`, `train_regressor.py` score model
- `encode_latents.py` VAE latent cache
- `model_fm.py` the U-Net
- `train_fm.py` flow matching training
- `generator.py` sampler, and the embedding to image model
- `select_images.py` best-of-N selection
- `export.py` writes and checks the submission files
- `validate.py` loads the final model
