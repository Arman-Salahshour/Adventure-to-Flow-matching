import quix
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

from constants import *


def load_data(datapath=DATAPATH):
    postprocess = (
        T.Compose([
            T.ToTensor(),
            T.Normalize(IN_MEAN, IN_STD),
        ]),
        T.ToTensor(),   # .scores.npy -> tensor
    )

    return quix.QuixDataset(
        'CarRecs',
        datapath,
        override_extensions=['jpg', 'scores.npy'],
    ).map_tuple(*postprocess)


def build_ratings(data, batch_size=64, num_workers=4):
    # No shuffling here, row i of the table must be data[i].
    loader = DataLoader(data, batch_size=batch_size, num_workers=num_workers)
    chunks = []
    for _, scores in loader:
        chunks.append(pd.DataFrame(scores.squeeze().numpy(), columns=RESPONDERS))
    ratings = pd.concat(chunks, ignore_index=True)
    return ratings.apply(pd.to_numeric, errors='coerce')


def split_indices(size, test_frac=0.1, seed=SEED):
    rng = np.random.default_rng(seed)
    idxs = rng.permutation(size)
    n_test = int(size * test_frac)
    return idxs[n_test:], idxs[:n_test]


def make_loaders(data, train_idx, test_idx, batch_size=64, num_workers=4):
    train_data = Subset(data, train_idx)
    test_data = Subset(data, test_idx)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


_mean = torch.tensor(IN_MEAN).view(1, 3, 1, 1)
_std = torch.tensor(IN_STD).view(1, 3, 1, 1)


def denorm(x):
    return (x * _std.to(x.device) + _mean.to(x.device)).clamp(0, 1)


def normalise(x):
    return (x - _mean.to(x.device)) / _std.to(x.device)
