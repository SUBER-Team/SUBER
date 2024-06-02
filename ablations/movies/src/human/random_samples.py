import typing
from environment import Simulatio4RecSys
from environment.reward_perturbator import NoPerturbator
from environment.items_retrieval import TimeItemsRetrieval
from environment.items_selection import GreedySelector
from environment.users.users_loader import UsersCSVLoader
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
import json
import tqdm


class MoviesRandomSampleHuman(AbstractCaseStudy):
    name = "movies_random_sample_human_hist"

    def __init__(self, create_env, run_name, is_open_ai=False) -> None:
        super().__init__(create_env=create_env, run_name=run_name)
        self.is_open_ai = is_open_ai

    def interact_sequential_ids(
        self,
        env,
        users_ids,
        previous_items_ids_ratings: list[tuple[int, float]],
        next_item_id: int,
    ):
        env.reset(seed=42)
        data = []
        rng = np.random.default_rng(42)
        for user in users_ids:
            env.reset(user_id=user)

            rng.shuffle(previous_items_ids_ratings)

            # access memory
            for a in previous_items_ids_ratings:
                item_id, r = a
                env.memory.update_memory(user, [item_id], [r])

            obs, reward, terminated, _, info = env.step(
                env.item_to_action[next_item_id]
            )
            data.append(
                pd.DataFrame(
                    {
                        "item": next_item_id,
                        "user": user,
                        "user_name": env.user_list[user].name,
                        "item_name": env.items_loader.load_items_from_ids(
                            [next_item_id]
                        )[0].title,
                        "LLM_explanation": info["LLM_explanation"],
                        "LLM_rating": info["LLM_rating"],
                        "LLM_interaction_HTML": info["LLM_interaction_HTML"],
                        "rating": reward,
                    },
                    index=[0],
                )
            )
        return pd.concat(data)

    def run(self):
        print(f"Running {self.name}")
        env = self.create_env()
        # Make report
        base_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            f"../../reports/{self.run_name}",
        )
        os.makedirs(base_path, exist_ok=True)
        env.rating_prompt.llm_query_explanation = True

        random_dataset = [
            {
                "user_id": 110,
                "positive": [157336, 13],
                "positive_rating": [10, 10],
                "negative": [274],
                "negative_rating": [6],
                "to_watch": 272,
            },  # Lily Chen
            {
                "user_id": 419,
                "positive": [245891, 694],
                "positive_rating": [9, 10],
                "negative": [62],
                "negative_rating": [6],
                "to_watch": 9486,
            },  # Ethan Taylor
            {
                "user_id": 565,
                "positive": [453, 27205],
                "positive_rating": [9, 10],
                "negative": [335984],
                "negative_rating": [5],
                "to_watch": 601,
            },  # Samuel Gomez
            {
                "user_id": 77,
                "positive": [85, 286217],
                "positive_rating": [9, 9],
                "negative": [91314],
                "negative_rating": [3],
                "to_watch": 8587,
            },  # James Thompson
        ]

        acc = []
        for d in tqdm.tqdm(random_dataset):
            user_id = d["user_id"]
            previous_items_ids_ratings = list(
                zip(d["positive"], d["positive_rating"])
            ) + list(zip(d["negative"], d["negative_rating"]))

            data = self.interact_sequential_ids(
                env, [user_id], previous_items_ids_ratings, d["to_watch"]
            )
            acc.append(data)

        data = pd.concat(acc)
        data["type"] = "random_sample"
        return data
