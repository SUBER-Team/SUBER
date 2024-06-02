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


# TODO pick 2 users from dataset, different one for - and +
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
            f"../../reports/{self.run_name}",
        )
        os.makedirs(base_path, exist_ok=True)

        env = self.create_env(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "../../datasets/movie_sagas_data.json",
            ),
            self.user_dataset,
        )
        env.rating_prompt.llm_query_explanation = True

        with open(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "../../datasets/movie_sagas_group.json",
            ),
        ) as json_file:
            sagas_ids = json.load(json_file)

        rng = np.random.default_rng(42)
        acc = []

        all_items_id = np.concatenate(sagas_ids)
        idx = rng.choice(range(len(sagas_ids)), size=2, replace=False)
        sagas_ids = [sagas_ids[i] for i in idx]

        for saga in tqdm.tqdm(sagas_ids):
            previos_items_ids = saga[:-1]
            next_item_id = saga[-1]

            other_ids = np.setdiff1d(all_items_id, saga)
            other_ids = rng.choice(other_ids, len(saga), replace=False)

            users_ids = rng.choice(env.num_users, size=1, replace=False)

            if len(acc) % 2 == 0:
                data_positive = self.interact_sequential_ids(
                    env,
                    users_ids,
                    previos_items_ids,
                    other_ids,
                    next_item_id,
                    positive=True,
                )
                acc.append(data_positive)
            else:
                data_negative = self.interact_sequential_ids(
                    env,
                    users_ids,
                    previos_items_ids,
                    other_ids,
                    next_item_id,
                    positive=False,
                )
                acc.append(data_negative)

        data = pd.concat(acc)

        data["type"] = "sagas"

        return data
