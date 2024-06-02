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


class MovieSagasHistoryRandomStudy(AbstractCaseStudy):
    name = "movie_sagas_random_history"

    def __init__(self, create_env, run_name, user_dataset, is_open_ai=False) -> None:
        super().__init__(create_env=create_env, run_name=run_name)
        self.is_open_ai = is_open_ai
        self.user_dataset = user_dataset

    def interact_sequential_ids(
        self,
        env,
        users_ids,
        previos_items_ids: typing.List[int],
        other_ids: typing.List[int],
        next_item_id: int,
        positive=True,
    ):
        env.reset(seed=42)
        data = []
        for user in users_ids:
            env.reset(user_id=user)

            interact_ids = np.concatenate([previos_items_ids, other_ids])
            np.random.shuffle(interact_ids)

            # access memory
            for item_id in interact_ids:
                r = env.items_loader.load_items_from_ids([item_id])[0].vote_average
                if r == 0.0:
                    r = 6.53  # TMDB est. average
                if item_id in previos_items_ids:
                    r = 10 if positive else 1
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
        base_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            f"../reports/{self.run_name}",
        )
        os.makedirs(base_path, exist_ok=True)

        env = self.create_env(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "../datasets/movie_sagas_data.json",
            ),
            self.user_dataset,
        )

        with open(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "../datasets/movie_sagas_group.json",
            ),
        ) as json_file:
            sagas_ids = json.load(json_file)

        NUM_SAMPLED_USERS = 100
        if self.is_open_ai:
            NUM_SAMPLED_USERS = 50

        acc_positive = []
        acc_negative = []

        acc_rate_positive = []
        acc_rate_negative = []

        all_items_id = np.concatenate(sagas_ids)

        for saga in tqdm.tqdm(sagas_ids):
            previos_items_ids = saga[:-1]
            next_item_id = saga[-1]

            other_ids = np.setdiff1d(all_items_id, saga)
            other_ids = np.random.choice(other_ids, len(saga), replace=False)

            users_ids = np.random.choice(
                env.num_users, NUM_SAMPLED_USERS, replace=False
            )

            data_positive = self.interact_sequential_ids(
                env,
                users_ids,
                previos_items_ids,
                other_ids,
                next_item_id,
                positive=True,
            )
            acc_positive.append(data_positive)

            data_negative = self.interact_sequential_ids(
                env,
                users_ids,
                previos_items_ids,
                other_ids,
                next_item_id,
                positive=False,
            )
            acc_negative.append(data_negative)

            ratings_positive = data_positive["LLM_rating"].values
            ratings_negative = data_negative["LLM_rating"].values

            perc_success_yes = (
                ratings_positive >= np.full_like(ratings_positive, 7)
            ).mean()
            perc_success_no = (
                ratings_negative <= np.full_like(ratings_negative, 5)
            ).mean()
            acc_rate_negative.append(perc_success_no)
            acc_rate_positive.append(perc_success_yes)

        perc_success_no = np.mean(acc_rate_negative)
        perc_success_yes = np.mean(acc_rate_positive)

        pd.DataFrame(
            {
                "Config": self.run_name,
                "Name": self.name,
                "Percentage success": (perc_success_no + perc_success_yes) / 2,
                "Percentage success (Positive)": perc_success_yes,
                "Percentage success (Negative)": perc_success_no,
            },
            index=[0],
        ).to_csv(f"{base_path}/{self.name}.csv", index=False)
