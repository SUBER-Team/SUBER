from environment import Simulatio4RecSys
from environment.reward_perturbator import NoPerturbator
from environment.items_retrieval import TimeItemsRetrieval
from environment.items_selection import GreedySelector
from ablations.utils import (
    AbstractCaseStudy,
    header_report,
    header_report_positive_negative,
    html_report,
    interact_sequential_ids,
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


user_genre_action = [
    User(
        "Emily Riga",
        "F",
        22,
        (
            "a spirited woman who thrives on adrenaline and excitement. With a penchant"
            " for action movies, she possesses an unwavering passion for thrilling"
            " plots, intense stunts, and explosive sequences. She embraces the"
            " adrenaline rush that these movies provide, making her an avid fan of the"
            " genre. She gives consistently high ratings (between 8-10) to action"
            " movies because she believes that this genre has the power to transport"
            " viewers into a realm of exhilaration and escapism. Action films allow her"
            " to experience a surge of energy and immerse herself in thrilling"
            " narratives. Moreover she disrespect every other genres and if the film's"
            " genres do not contains action she give a low rating (between 1-5)."
        ),
    )
]


user_genre_comedy = [
    User(
        "Alex Wallace",
        "M",
        22,
        (
            "a witty and sarcastic individual, is a self-proclaimed connoisseur of"
            " comedy movies. With a sharp sense of humor and a knack for comedic"
            " timing, Alex has an extensive collection of comedy films and quotes"
            " memorized. Their quick wit and ability to find humor in any situation"
            " make them a sought-after companion for movie nights. Their high ratings"
            " (between 8-10) for comedy movies stem from their appreciation of clever"
            " writing, comedic timing, and the ability of comedies to provide moments"
            " of genuine laughter. They thoroughly enjoy the genre's ability to poke"
            " fun at life's absurdities and make everyday situations hilarious."
            " Conversely, if a movie lacks comedic elements, Alex's disappointment"
            " leads them to assign a low rating (between 1-5), as they strongly prefer"
            " movies that can evoke laughter and amusement."
        ),
    )
]

# TODO: run one with + and one with -


class GenrePreferencePaperStudy(AbstractCaseStudy):
    name = "genre_preference_paper"

    def __init__(self, create_env, run_name, max_genres=2) -> None:
        super().__init__(create_env=create_env, run_name=run_name)
        self.max_genres = max_genres

    def _get_env(self, users, movies):
        env = self.create_env(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                movies,
            ),
            UsersListLoader(users),
        )
        env.rating_prompt.llm_query_explanation = True
        return env

    def run(self):
        print(f"Running {self.name}")
        base_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            f"../../reports/{self.run_name}",
        )
        os.makedirs(base_path, exist_ok=True)

        configs = {
            "Action": user_genre_action,
            "Comedy": user_genre_comedy,
        }

        data_no_acc = []
        data_yes_acc = []
        for genre, users in tqdm.tqdm(list(configs.items())[: self.max_genres]):
            # Set environment
            genre_lower = genre.lower()
            env_yes = self._get_env(
                users,
                f"../../datasets/genres_20/{genre_lower}.json",
            )

            rng = np.random.default_rng(42)

            random_yes = rng.choice(
                a=range(len(env_yes.item_ids)),
            )

            data_yes, _ = interact_sequential_ids(env_yes, [0], [random_yes])

            env_no = self._get_env(
                users,
                f"../../datasets/genres_20/no_{genre_lower}.json",
            )

            random_no = rng.choice(a=range(len(env_no.item_ids)))
            data_no, _ = interact_sequential_ids(env_no, [0], [random_no])

            data_no_acc.append(data_no)
            data_yes_acc.append(data_yes)

        data = [*data_no_acc, *data_yes_acc]
        data = pd.concat(data)
        data["type"] = "genre_preference"

        return data
