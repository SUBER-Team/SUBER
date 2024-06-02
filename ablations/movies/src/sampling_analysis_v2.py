# %%
import os
from scipy.stats import norm
import numpy as np
import pandas as pd


# %%
# KL between 2 gaussian distributions, function
def kl_divergence(mu1, sigma1, mu2, sigma2):
    return (
        np.log(sigma2 / sigma1)
        + (sigma1**2 + (mu1 - mu2) ** 2) / (2 * sigma2**2)
        - 0.5
    )


def p(x, ratings):
    return len(ratings[ratings == x]) / len(ratings) + 1e-6


# %%
def run_config(name, in_file):
    name = name.split("/")[-1]
    data = pd.read_csv(in_file)

    data["rating"].hist(bins=10)

    movielens_ratings = pd.read_csv(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../../../environment/movies/datasets/ml-latest-small/ratings.csv",
        )
    )
    movielens_ratings["rating"] = movielens_ratings["rating"] * 2

    probs_our = np.array(
        [
            len(data["rating"][data["rating"] == x]) / len(data["rating"])
            for x in range(1, 11)
        ]
    )
    probs_movielens = np.array(
        [
            0.022166718841458488,
            0.038346176907715024,
            0.034127715161148334,
            0.09881372156302684,
            0.07641544304527774,
            0.19559479666320065,
            0.14796326375812152,
            0.22659525034875633,
            0.07552053618598108,
            0.08445637752532151,
        ]
    )
    global_total_variation_distance = 0.5 * np.sum(np.abs(probs_our - probs_movielens))

    return pd.DataFrame(
        {
            "Config": name,
            "Name": "Sampling",
            "global_total_variation_distance": global_total_variation_distance,
            **data["rating"].describe().to_dict(),
        },
        index=[0],
    )


# %%
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--split", type=str, default="sampling_distribution")
args = parser.parse_args()
path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "../",
    args.name,
    f"{args.split}_v2.csv",
)

in_file = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "../",
    args.name,
    f"{args.split}_rating_dump.csv",
)
if os.path.exists(in_file):
    df = run_config(args.name, in_file)
    df.to_csv(path, index=False)
