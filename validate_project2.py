import os
import sys

sys.path.insert(0, 'src')

from constants import *

CKPT = os.path.join(OUTDIR, 'final_model.pth')


def load_my_models():
    """Returns the final generative model with weights loaded."""
    # TODO: swap in the generator once it exists.
    raise NotImplementedError('generator not built yet')


if __name__ == '__main__':
    m = load_my_models()
    print(type(m).__name__, 'loaded')
