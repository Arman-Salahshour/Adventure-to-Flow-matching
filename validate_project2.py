import os
import sys

# Works from any directory, including when the graders import it
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from constants import *
from generator import CarGenerator

CKPT = os.path.join(OUTDIR, 'final_model.pth')


def load_my_models():
    """Returns the final generative model with weights loaded.

    Each row of embeddings.pth is [z, c]: z is the starting noise (768 values),
    c the six scores (0-9). model(emb) or model.decode(z, c) gives the image,
    in the same ImageNet-normalised space as the dataset loader.
    The VAE is the public pretrained one and is fetched from Hugging Face.
    """
    return CarGenerator.from_checkpoint(CKPT)


if __name__ == '__main__':
    m = load_my_models()
    print(type(m).__name__, 'loaded')
