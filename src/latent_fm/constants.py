import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.nn as nn

# This folder, and the repo root two levels up; paths work from any working directory
PKG = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(PKG))

# Packages installed by `make deps` go last on the path,
# so they never shadow what the venv already has
sys.path.append(os.path.join(PKG, '.deps'))

SEED = 1234
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# The same embedding must always give the same image
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DATAPATH = '/projects/ec674/data/'
# Own subfolder, so this pipeline never overwrites the main deliverable in output/
OUTDIR = os.path.join(ROOT, 'output', 'latent_fm')

IN_MEAN = [0.485, 0.456, 0.406]
IN_STD = [0.229, 0.224, 0.225]

RESPONDERS = ['Raouf', 'Moira', 'Louie', 'Ferdinando', 'Gragar', 'Esther']
MOIRA = 1
FERDINANDO = 3

MAX_SCORE = 9.0
IMG_SHAPE = (3, 96, 128)

# Frozen autoencoder and the latent it produces
VAE_NAME = 'stabilityai/sd-vae-ft-mse'
LATENT_SHAPE = (4, 12, 16)
LATENT_DIM = 4 * 12 * 16
