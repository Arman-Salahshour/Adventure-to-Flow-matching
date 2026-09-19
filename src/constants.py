import torch
import numpy as np
import pandas as pd
import torch.nn as nn

SEED = 1234
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATAPATH = '/projects/ec674/data/'
OUTDIR = 'output'

IN_MEAN = [0.485, 0.456, 0.406]
IN_STD = [0.229, 0.224, 0.225]

RESPONDERS = ['Raouf', 'Moira', 'Louie', 'Ferdinando', 'Gragar', 'Esther']
MOIRA = 1
FERDINANDO = 3

MAX_SCORE = 9.0
IMG_SHAPE = (3, 96, 128)
