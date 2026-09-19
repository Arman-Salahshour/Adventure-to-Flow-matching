import os
from torchvision import models
from scipy.stats import spearmanr

from constants import *


def make_regressor(pretrained=True, freeze_backbone=False):
    weights = 'IMAGENET1K_V1' if pretrained else None
    model = models.resnet18(weights=weights)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(RESPONDERS))
    return model


def constant_baseline(train_scores, test_scores):
    # MSE of always predicting each rater's training mean.
    mu = train_scores.mean(0) / MAX_SCORE
    return ((test_scores / MAX_SCORE - mu)**2).mean().item()


def train_regressor(model, train_loader, test_loader, epochs=30, lr=1e-4, ckpt=None, verbose=True):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    best = float('inf')
    history = []

    for epoch in range(epochs):
        model.train()
        for img, sc in train_loader:
            img = img.to(device)
            sc = sc.reshape(-1, 6).float().to(device) / MAX_SCORE
            opt.zero_grad()
            loss = crit(model(img), sc)
            loss.backward()
            opt.step()

        model.eval()
        tot = n = 0
        with torch.no_grad():
            for img, sc in test_loader:
                img = img.to(device)
                sc = sc.reshape(-1, 6).float().to(device) / MAX_SCORE
                tot += crit(model(img), sc).item() * len(img)
                n += len(img)
        val = tot / n
        history.append(val)

        if val < best:
            best = val
            if ckpt is not None:
                torch.save(model.state_dict(), ckpt)
        if verbose:
            print(f'{epoch:3d} val {val:.5f}')

    return history


def predict(model, loader):
    # Returns predictions and targets in 0-9 units.
    model = model.to(device).eval()
    P, T = [], []
    with torch.no_grad():
        for img, sc in loader:
            P.append(model(img.to(device)).cpu() * MAX_SCORE)
            T.append(sc.reshape(-1, 6).float())
    return torch.cat(P).numpy(), torch.cat(T).numpy()


def evaluate(P, T, mask=None, verbose=True):
    rows = []
    for j, nm in enumerate(RESPONDERS):
        t, p = T[:, j], P[:, j]
        if mask is not None:
            keep = mask(t)
            t, p = t[keep], p[keep]
        r2 = 1 - ((t - p)**2).sum() / ((t - t.mean())**2).sum()
        rho = spearmanr(t, p).statistic
        mae = abs(t - p).mean()
        rows.append({'rater': nm, 'n': len(t), 'r2': r2, 'rho': rho, 'mae': mae})
        if verbose:
            print(f'{nm} n={len(t)} r2={r2:.3f} rho={rho:.3f} mae={mae:.2f}')

    if verbose:
        print('pred corr M/F:', spearmanr(P[:, MOIRA], P[:, FERDINANDO]).statistic)
    return pd.DataFrame(rows)


def load_ensemble(paths):
    ens = []
    for p in paths:
        m = make_regressor(pretrained=False)
        m.load_state_dict(torch.load(p, map_location=device))
        ens.append(m.to(device).eval().requires_grad_(False))
    return ens


def ensemble_scores(ens, img):
    # Pessimistic aggregate: the lowest score any member gives.
    s = torch.stack([m(img) for m in ens])
    return s.min(dim=0).values * MAX_SCORE
