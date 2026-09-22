"""
Samples candidates for Moira, keeps the best 10 and exports them.

Best-of-N: draw many cars conditioned on Moira = 9, score them with the
regressor ensemble, drop near-copies of real cars, keep 10 that are good
and different from each other.
"""
import os
import sys
import json
from matplotlib import pyplot as plt

from constants import *
from generator import build_generator
from regressor import load_ensemble, ensemble_scores

sys.path.insert(0, ROOT)
from export import export

N_CANDIDATES = 512
BATCH = 64
N_KEEP = 10
N_CONTROL = 128

LATENTS = os.path.join(OUTDIR, 'latents.pth')
EMA = os.path.join(OUTDIR, 'fm_ema.pth')
DONE = [os.path.join(OUTDIR, f) for f in ('images.pth', 'embeddings.pth', 'final_model.pth')]


def moira_target(scores_train):
    # Others at their mean over the cars Moira liked most, so the whole
    # vector looks like a real car she rated high; then Moira itself at 9
    top = scores_train[:, MOIRA] >= scores_train[:, MOIRA].quantile(0.9)
    c = scores_train[top].mean(0)
    c[MOIRA] = MAX_SCORE
    return c


def generate(gen, ens, z, c):
    lat, img, pred = list(), list(), list()
    for i in range(0, len(z), BATCH):
        x1 = gen.latent(z[i:i + BATCH], c[i:i + BATCH])
        im = gen.decode_latent(x1)
        with torch.no_grad():
            p = ensemble_scores(ens, im)
        lat.append(x1)
        img.append(im)
        pred.append(p)
    return torch.concat(lat), torch.concat(img), torch.concat(pred)


def nn_dist(a, b):
    # Distance from each row of a to its nearest row of b
    return torch.cdist(a.flatten(1), b.flatten(1)).min(dim=1).values


def main():
    if all(os.path.exists(f) for f in DONE):
        print('samples: submission files found, skipping')
        return

    latent_dict = torch.load(LATENTS, map_location=device, weights_only=False)
    real_train = latent_dict['latents_train'].to(device)
    real_test = latent_dict['latents_test'].to(device)
    real_all = torch.concat([real_train, real_test])
    scores_train = latent_dict['scores_train']

    gen = build_generator(EMA, LATENTS)
    paths = [os.path.join(OUTDIR, f'reg_seed{s}.pth') for s in range(3)]
    ens = load_ensemble([p for p in paths if os.path.exists(p)])
    print(f'samples: {len(ens)} regressor(s) in the ensemble')

    target = moira_target(scores_train).to(device)
    g = torch.Generator(device=device).manual_seed(SEED)
    z = torch.randn(N_CANDIDATES, LATENT_DIM, device=device, generator=g)
    c = target.repeat(N_CANDIDATES, 1)

    lat, img, pred = generate(gen, ens, z, c)
    moira = pred[:, MOIRA]

    # Novelty: a real test car's nearest training car sets the scale;
    # a candidate closer than 90% of those is treated as a near-copy
    real_gap = nn_dist(real_test, real_train)
    copy_thresh = real_gap.quantile(0.1)
    cand_gap = nn_dist(lat, real_all)
    novel = cand_gap >= copy_thresh

    # Greedy: best first, skip anything too close to one already kept
    div_thresh = 0.5 * real_gap.median()
    picked = list()
    for i in moira.argsort(descending=True).tolist():
        if not novel[i]:
            continue
        if picked and nn_dist(lat[i:i + 1], lat[picked]).item() < div_thresh:
            continue
        picked.append(i)
        if len(picked) == N_KEEP:
            break

    # Fallbacks, only if the filters left too few
    for i in moira.argsort(descending=True).tolist():
        if len(picked) == N_KEEP:
            break
        if i not in picked and novel[i]:
            picked.append(i)
    for i in moira.argsort(descending=True).tolist():
        if len(picked) == N_KEEP:
            break
        if i not in picked:
            picked.append(i)

    # Conditioning check: same noise at Moira = 0 and Moira = 9
    c0 = target.repeat(N_CONTROL, 1)
    c0[:, MOIRA] = 0
    _, _, pred0 = generate(gen, ens, z[:N_CONTROL], c0)

    stats = {
        'n_candidates': N_CANDIDATES,
        'guidance': gen.guidance,
        'euler_steps': gen.steps,
        'target_scores': target.tolist(),
        'pred_moira_all_candidates': moira.mean().item(),
        'pred_moira_selected': moira[picked].mean().item(),
        'pred_moira_control_cond0': pred0[:, MOIRA].mean().item(),
        'pred_moira_control_cond9': moira[:N_CONTROL].mean().item(),
        'real_moira_mean_train': scores_train[:, MOIRA].mean().item(),
        'novel_fraction': novel.float().mean().item(),
        'copy_threshold': copy_thresh.item(),
        'selected_nn_dist': cand_gap[picked].tolist(),
    }
    with open(os.path.join(OUTDIR, 'selection_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    pd.DataFrame({
        'pred_moira': moira.cpu().numpy(),
        'nn_dist': cand_gap.cpu().numpy(),
        'novel': novel.cpu().numpy(),
        'picked': [i in picked for i in range(N_CANDIDATES)],
    }).to_csv(os.path.join(OUTDIR, 'candidates.csv'), index=False)

    # Decode the final 10 again as one batch; this is what gets checked
    images, emb = export(gen, z[picked], c[picked])

    with torch.no_grad():
        final_pred = ensemble_scores(ens, images.to(device))[:, MOIRA]
    print('samples: predicted Moira of the final 10:', [round(p, 2) for p in final_pred.tolist()])
    for k, v in stats.items():
        if not isinstance(v, list):
            print(f'  {k}: {v}')

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    mean = torch.tensor(IN_MEAN).view(3, 1, 1)
    std = torch.tensor(IN_STD).view(3, 1, 1)
    for k, ax in enumerate(axes.flat):
        ax.imshow((images[k] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy())
        ax.set_title(f'M {final_pred[k].item():.1f}', fontsize=9)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'final_images.png'), dpi=150)


if __name__ == '__main__':
    main()
