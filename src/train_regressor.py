import os

from constants import *
from data import load_data, build_ratings, split_indices, make_loaders
from regressor import make_regressor, train_regressor, constant_baseline, predict, evaluate

SEEDS = [0, 1, 2]
EPOCHS = 30


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    data = load_data()
    ratings = build_ratings(data)
    train_idx, test_idx = split_indices(len(data))
    train_loader, test_loader = make_loaders(data, train_idx, test_idx)

    scores_all = torch.tensor(ratings.values, dtype=torch.float32)
    base = constant_baseline(scores_all[train_idx], scores_all[test_idx])
    print(f'constant baseline mse {base:.5f}')

    # Same split for every seed, only the init and batch order change.
    for s in SEEDS:
        torch.manual_seed(SEED + s)
        ckpt = os.path.join(OUTDIR, f'reg_seed{s}.pth')
        print(f'--- seed {s} ---')
        model = make_regressor()
        train_regressor(model, train_loader, test_loader, epochs=EPOCHS, ckpt=ckpt)

        P, T = predict(model, test_loader)
        evaluate(P, T)
        print('tail (true >= 7):')
        evaluate(P, T, mask=lambda t: t >= 7)


if __name__ == '__main__':
    main()
