"""Writes the three submission artifacts and checks they are consistent.

The graders pass embeddings.pth back through the model and compare against
images.pth, so decoding has to be deterministic and the ordering has to match.
"""
import os
import sys

sys.path.insert(0, 'src')

from constants import *

N_IMAGES = 10
TOL = 1e-5


def export(model, z, c=None, outdir=OUTDIR):
    """model: object with .decode(z, c); z: [10, D]; c: [10, 6] conditioning."""
    os.makedirs(outdir, exist_ok=True)
    model = model.to(device).eval()
    z = z.to(device)
    c = None if c is None else c.to(device)

    with torch.no_grad():
        images = model.decode(z, c) if c is not None else model.decode(z)
    images = images.cpu()

    assert images.shape == (N_IMAGES, *IMG_SHAPE), images.shape

    # Conditioning travels with the latent so the embedding fully determines
    # the image.
    embeddings = z.cpu() if c is None else torch.cat([z.cpu(), c.cpu()], dim=1)

    torch.save(images, os.path.join(outdir, 'images.pth'))
    torch.save(embeddings, os.path.join(outdir, 'embeddings.pth'))
    torch.save(model.state_dict(), os.path.join(outdir, 'final_model.pth'))

    check(model, outdir)
    print(f'wrote {N_IMAGES} images, embedding dim {embeddings.shape[1]}')
    return images, embeddings


def check(model, outdir=OUTDIR, latent_dim=None):
    images = torch.load(os.path.join(outdir, 'images.pth'), map_location=device)
    emb = torch.load(os.path.join(outdir, 'embeddings.pth'), map_location=device)

    d = emb.shape[1] - 6 if latent_dim is None else latent_dim
    z, c = emb[:, :d], emb[:, d:]

    with torch.no_grad():
        redone = model.to(device).eval().decode(z, c)

    err = (redone - images).abs().max().item()
    assert err < TOL, f'round trip failed, max abs diff {err}'
    print(f'round trip ok, max abs diff {err:.2e}')


if __name__ == '__main__':
    from validate_project2 import load_my_models
    check(load_my_models())
