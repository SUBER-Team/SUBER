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
)
import numpy as np
from environment.users import User, UsersListLoader
import os
import pandas as pd
import tqdm


def interact_sequential_ids(env, users_ids, items_ids, tqdm_disabled=True):
    num_users = env.num_users
    num_items = env.num_items
    env.reset(seed=42)

    vote_average_tmdb = np.zeros(shape=(num_items))
    for i in items_ids:
        vote_average_tmdb[i] = env.items_loader.load_items_from_ids(
            [env.action_to_item[i]]
        )[0].vote_average

    data = []

    for user in tqdm.tqdm(users_ids, disable=tqdm_disabled):
        env.reset(user_id=user)
        for item in items_ids:
            obs, reward, terminated, _, info = env.step(item)
            data.append(
                pd.DataFrame(
                    {
                        "item": item,
                        "user": user,
                        "user_name": env.user_list[user].name,
                        "item_name": env.items_loader.load_items_from_ids(
                            [env.action_to_item[item]]
                        )[0].title,
                        "LLM_explanation": info["LLM_explanation"],
                        "LLM_rating": info["LLM_rating"],
                        "LLM_interaction_HTML": info["LLM_interaction_HTML"],
                        "rating": reward,
                    },
                    index=[0],
                )
            )
    return pd.concat(data), vote_average_tmdb


class SamplingExplanationPaper(AbstractCaseStudy):
    name = "sampling_explanation_paper"

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

        NUM_SAMPLED_USERS = 2
        NUM_SAMPLED_MOVIES = 11

        users_ids = [
            # 384,
            # 113,
            145,
            # 147,
        ]  # np.random.choice(env.num_users, NUM_SAMPLED_USERS, replace=False)

        acc = []
        for user_id in tqdm.tqdm(users_ids):
            # item_ids = np.random.choice(
            #     env.num_users, NUM_SAMPLED_MOVIES, replace=False
            # )
            item_ids = [
                348350,
                10071,
                63,
                81003,
                173,
                744,
                14458,
                9675,
                11,
                124680,
            ]
            item_ids = [env.item_to_action[i] for i in item_ids]

            data, vote_average_tmdb = interact_sequential_ids(env, [user_id], item_ids)
            data["TMDB_id"] = data["item"].apply(lambda x: env.action_to_item[x])
            acc.append(data)
        data = pd.concat(acc)
        data.to_csv(f"{base_path}/{self.name}_dump.csv", index=False)
