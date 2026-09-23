from scipy.stats import spearmanr

from constants import *


def correlation_matrix(ratings):
    rho = spearmanr(ratings.values).statistic
    return pd.DataFrame(rho, index=RESPONDERS, columns=RESPONDERS)


def top_decile(ratings, name, q=0.9):
    # Scores are integers, so the cut lands on a tied value and the set is
    # rarely exactly 10% of the data.
    cut = ratings[name].quantile(q)
    return set(ratings[ratings[name] >= cut].index)


def overlap_lift(ratings, a, b, q=0.9):
    top_a = top_decile(ratings, a, q)
    top_b = top_decile(ratings, b, q)
    shared = top_a & top_b
    expected = len(top_a) * len(top_b) / len(ratings)
    return {
        'n_a': len(top_a),
        'n_b': len(top_b),
        'shared': len(shared),
        'expected': expected,
        'lift': len(shared) / expected,
        'indices': shared,
    }
