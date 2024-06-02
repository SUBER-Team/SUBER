from environment import Simulatio4RecSys
from environment.reward_perturbator import NoPerturbator
from environment.items_retrieval import TimeItemsRetrieval
from environment.items_selection import GreedySelector
from ablations.utils import (
    AbstractCaseStudy,
    header_report,
    header_report_positive_negative,
    html_report,
    interact_sequential,
    plot_heatmap,
    plot_heatmap_2_sides,
    plot_tmdb_corr,
    plot_users,
    data_to_matrix,
    interact_sequential_ids,
)
import numpy as np
from environment.users import User, UsersListLoader
import os
import pandas as pd
import tqdm


class SamplingExplanationStudy(AbstractCaseStudy):
    name = "sampling_explanation_paper_search"

    def __init__(self, create_env, run_name) -> None:
        super().__init__(create_env=create_env, run_name=run_name)

    def run(self):
        print(f"Running {self.name}")
        env = self.create_env()
        # Make report
        base_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            f"../reports/{self.run_name}",
        )
        os.makedirs(base_path, exist_ok=True)
        env.rating_prompt.llm_query_explanation = True

        NUM_SAMPLED_USERS = 50
        NUM_SAMPLED_MOVIES = 10

        users_ids = np.random.choice(env.num_users, NUM_SAMPLED_USERS, replace=False)

        acc = []
        for user_id in tqdm.tqdm(users_ids):
            item_ids = np.random.choice(
                env.num_users, NUM_SAMPLED_MOVIES, replace=False
            )
            data, vote_average_tmdb = interact_sequential_ids(env, [user_id], item_ids)
            data["TMDB_id"] = data["item"].apply(lambda x: env.action_to_item[x])
            acc.append(data)
        data = pd.concat(acc)
        data.to_csv(f"{base_path}/{self.name}_dump.csv", index=False)
