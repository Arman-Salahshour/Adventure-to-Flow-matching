from matplotlib import pyplot as plt

from constants import *
from data import denorm


def show(data, indices, title, ncol=10, nrow=2):
    idx = list(indices)[:ncol * nrow]
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.6, nrow * 2))
    for ax, i in zip(axes.flat, idx):
        img, s = data[i]
        ax.imshow(denorm(img[None])[0].permute(1, 2, 0).numpy())
        ax.set_title(f"M{int(s.flatten()[MOIRA])} F{int(s.flatten()[FERDINANDO])}", fontsize=7)
        ax.axis('off')
    for ax in axes.flat[len(idx):]:
        ax.axis('off')
    fig.suptitle(title)
    plt.tight_layout()
    return fig


def show_tensor(imgs, title, ncol=5, captions=None):
    # imgs are already normalised, shape [N, 3, 96, 128].
    n = len(imgs)
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.6, nrow * 2))
    for k, ax in enumerate(axes.flat):
        if k < n:
            ax.imshow(denorm(imgs[k:k + 1].cpu())[0].permute(1, 2, 0).numpy())
            if captions is not None:
                ax.set_title(captions[k], fontsize=7)
        ax.axis('off')
    fig.suptitle(title)
    plt.tight_layout()
    return fig


def rating_histograms(ratings):
    ratings.hist(figsize=(12, 8), bins=9, edgecolor='black', color='skyblue')
    plt.suptitle('Distribution of Image Ratings by Each Rater', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return plt.gcf()
