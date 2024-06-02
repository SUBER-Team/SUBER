import os

import numpy as np
import pandas as pd
import tqdm

from ablations.utils import (
    AbstractCaseStudy,
    data_to_matrix,
    header_report,
    header_report_positive_negative,
    html_report,
    interact_sequential,
    interact_sequential_ids,
    plot_heatmap,
    plot_heatmap_2_sides,
    plot_tmdb_corr,
    plot_users,
)
from environment import Simulatio4RecSys
from environment.items_retrieval import TimeItemsRetrieval
from environment.items_selection import GreedySelector
from environment.reward_perturbator import NoPerturbator
from environment.users import User, UsersListLoader


class SamplingSubsetInteractionsStudy(AbstractCaseStudy):
    name = "sampling_distribution"

    def __init__(self, create_env, run_name, is_open_ai=False) -> None:
        super().__init__(create_env=create_env, run_name=run_name)
        self.is_open_ai = is_open_ai

    def run(self):
        print(f"Running {self.name}")
        env = self.create_env()
        # Make report
        base_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            f"../reports/{self.run_name}",
        )
        os.makedirs(base_path, exist_ok=True)

        NUM_SAMPLED_INTERATCTIONS = 100 * 20 if self.is_open_ai else 100 * 1000

        acc = []
        for _ in tqdm.tqdm(range(NUM_SAMPLED_INTERATCTIONS)):
            users_ids = np.random.choice(env.num_users, 1, replace=False)
            item_ids = np.random.choice(env.num_items, 1, replace=False)
            data, vote_average_tmdb = interact_sequential_ids(env, users_ids, item_ids)
            data["book_id"] = data["item"].apply(lambda x: env.action_to_item[x])
            acc.append(data)
        data = pd.concat(acc)
        data.to_csv(f"{base_path}/{self.name}_rating_dump.csv", index=False)
