import argparse
import os

import pandas as pd

import environment.LLM as LLM
from ablations.movies.src.paper_examples_showcase_sampling import (
    SamplingExplanationPaper,
)
from environment import Simulatio4RecSys
from environment.items_selection import GreedySelector
from environment.movies.configs import (
    OPTIONS_ITEMS_RETRIEVAL,
    OPTIONS_LLM_RATER,
    OPTIONS_REWARD_PERTURBATOR,
    get_base_parser,
    get_enviroment_from_args,
    get_items_retrieval,
    get_llm_rater,
    get_reward_perturbator,
    get_user_dataset,
)
from environment.movies.movies_loader import MoviesLoader
from environment.reward_shaping import IdentityRewardShaping
from environment.users import UsersLoader

from .genre_preference_paper import GenrePreferencePaperStudy
from .movie_sagas_history_random import MovieSagasHistoryRandomStudy
from .random_samples import MoviesRandomSampleHuman


def parse_args():
    parser = get_base_parser()
    parser.add_argument("--random-rating", action="store_true", default=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    name = args.llm_model

    model = LLM.load_LLM(name)

    # During the ablation study we want to look at the raw ratings, not the reward for RL
    args.reward_shaping = "identity"
    name_report = name.replace("/", "_")
    exp_name = (
        f"{name_report}-{args.llm_rater}-{args.perturbator}-{args.items_retrieval}"
    )
    if args.user_dataset == "basic":
        exp_name += "-basic_users"
    if args.random_rating:
        name_report = f"{name_report}-random_rating"

    rater = get_llm_rater(args.llm_rater, model, args.items_retrieval != "none")
    rater.random_rating = args.random_rating

    def create_env(item: str, user_loader: UsersLoader):
        return Simulatio4RecSys(
            render_mode=None,
            items_loader=MoviesLoader(item),
            users_loader=user_loader,
            items_selector=GreedySelector(),
            reward_perturbator=get_reward_perturbator(args.perturbator, seed=42),
            items_retrieval=get_items_retrieval(args.items_retrieval),
            llm_rater=rater,
            reward_shaping=IdentityRewardShaping(seed=42),
        )

    # Genre preference paper
    acc = []
    # genre_preference_study = GenrePreferencePaperStudy(create_env, exp_name)
    # res = genre_preference_study.run()
    # acc.append(res)

    # # Movie sagas history random
    # movie_sagas_history_random_study = MovieSagasHistoryRandomStudy(
    #     create_env, exp_name, user_dataset=get_user_dataset(args.user_dataset)
    # )
    # res = movie_sagas_history_random_study.run()
    # acc.append(res)

    # Random samples
    random_samples = MoviesRandomSampleHuman(
        lambda: get_enviroment_from_args(model, args), exp_name
    )
    res = random_samples.run()
    acc.append(res)

    data = pd.concat(acc)

    base_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        f"../../reports_human/{name_report}",
    )
    os.makedirs(base_path, exist_ok=True)

    data.to_csv(os.path.join(base_path, f"samples.csv"))
