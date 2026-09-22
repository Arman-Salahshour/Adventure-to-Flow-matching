"""Encodes every car with the frozen SD VAE and caches the latents."""
import os
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader

from constants import *
from data import load_data, split_indices, denorm
from diffusers import AutoencoderKL

LATENTS = os.path.join(OUTDIR, 'latents.pth')


def encode(vae, loader):
    latents_all = list()
    scores_all = list()
    with torch.no_grad():
        for img, scores in tqdm(loader):
            # VAE expects [-1, 1], the loader gives ImageNet-normalised images
            img = (denorm(img) * 2 - 1).to(device)
            scores = scores.reshape(-1, 6).float()
            # Mean rather than a sample, so each car has one fixed latent
            latents = vae.encode(img).latent_dist.mean * vae.config.scaling_factor

            latents_all.append(latents.cpu())
            scores_all.append(scores)

    return torch.concat(latents_all, dim=0), torch.concat(scores_all, dim=0)


def main():
    if os.path.exists(LATENTS):
        print('latents: found, skipping')
        return
    os.makedirs(OUTDIR, exist_ok=True)

    data = load_data()
    train_idx, test_idx = split_indices(len(data))

    vae = AutoencoderKL.from_pretrained(VAE_NAME)
    vae = vae.to(device).eval()

    # No shuffling: row i of the cache is data[train_idx[i]]
    train_loader = DataLoader(Subset(data, train_idx), batch_size=64, shuffle=False)
    test_loader = DataLoader(Subset(data, test_idx), batch_size=64, shuffle=False)

    latents_train, scores_train = encode(vae, train_loader)
    latents_test, scores_test = encode(vae, test_loader)

    torch.save({
        'latents_train': latents_train,
        'latents_test': latents_test,
        'scores_train': scores_train,
        'scores_test': scores_test,
        'train_idx': train_idx,
        'test_idx': test_idx,
        'scaling_factor': vae.config.scaling_factor,
    }, LATENTS)
    print(f'latents: saved {len(latents_train)} train, {len(latents_test)} test')


if __name__ == '__main__':
    main()
