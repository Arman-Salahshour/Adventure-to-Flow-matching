# Project 2, Moira's Dream Car

IN5310, master's task: 10 cars optimal for Moira.

Pipeline: a frozen SD VAE maps cars to 4x12x16 latents, a conditional flow
matching U-Net learns to generate latents from the six scores, and a ResNet-18
ensemble picks the best candidates at Moira = 9.

## Run

With the course venv active, from the repo root:

    make

Each step skips work whose output already exists in `output/`, so a rerun only
does what is missing. Steps can also run on their own:

| target | does | writes |
| --- | --- | --- |
| `make deps` | installs `diffusers` into `.deps` if missing | `.deps/` |
| `make regressor` | trains missing regressor seeds | `reg_seed{0,1,2}.pth` |
| `make latents` | encodes all cars with the VAE | `latents.pth` |
| `make generator` | trains the flow matching U-Net | `fm_ema.pth`, `fm_log.csv` |
| `make samples` | picks and exports the final 10 | `images.pth`, `embeddings.pth`, `final_model.pth` |
| `make validate` | loads the model and re-decodes the embeddings | |
| `make clean` | removes only the submission files | |

Training takes about an hour on one GPU, so run it in the background:

    mkdir -p logs && sbatch run.slurm

## Layout

- `src/constants.py` paths, seed, rater names
- `src/data.py` quix loading, split, normalisation
- `src/eda.py`, `src/viz.py` preference analysis and plots
- `src/regressor.py`, `src/train_regressor.py` score model
- `src/encode_latents.py` VAE latent cache
- `src/model_fm.py` the U-Net
- `src/train_fm.py` flow matching training
- `src/generator.py` sampler, and the embedding to image model
- `src/select_images.py` best-of-N selection and export
- `export.py` writes and checks the submission files
- `validate_project2.py` `load_my_models()` for the teaching team
- `lab.ipynb` working notebook
- `EVALUATION.md` state of the generator

## Embeddings

Each row of `embeddings.pth` is 774 numbers: the 768-value starting noise,
then the six scores in 0 to 9. `model(emb)` gives the image. `images.pth` is
in the same ImageNet-normalised space as the dataset loader.

## Notes

- `load_my_models()` fetches the pretrained VAE from Hugging Face, so it needs
  network access the first time.
- `final_model.pth` holds only the trained U-Net, about 131 MB.
