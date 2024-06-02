import argparse
import os

from environment import LLM
from environment.movies.movies_loader import MoviesLoader
from ..env import Simulatio4RecSys
from ..users import UsersCSVLoader
from ..items_retrieval import (
    SentenceSimilarityItemsRetrieval,
    SimpleMoviesRetrieval,
    TimeItemsRetrieval,
)
from ..items_selection import GreedySelector
from ..reward_perturbator import GaussianPerturbator, GreedyPerturbator, NoPerturbator
from .rater_prompts.our_system_prompt import (
    ThirdPersonDescriptive09_2Shot_OurSys,
    ThirdPersonDescriptive09_OurSys,
    ThirdPersonDescriptive09_1Shot_OurSys,
    ThirdPersonDescriptive110_2Shot_OurSys,
    ThirdPersonDescriptive110_1Shot_OurSys,
    ThirdPersonDescriptiveOneTen_2Shot_OurSys,
    ThirdPersonDescriptiveOneTen_1Shot_OurSys,
    ThirdPersonDescriptive09_2Shot_OurSys,
    ThirdPersonDescriptive110_OurSys,
)
from .rater_prompts import (
    ThirdPersonDescriptive09,
    ThirdPersonDescriptive09_1Shot,
    ThirdPersonDescriptive09_2Shot,
)
from environment.reward_shaping import (
    RewardReshapingExpDecayTime,
    RewardReshapingRandomWatch,
    IdentityRewardShaping,
    RewardReshapingTerminateIfSeen,
)

from gymnasium.utils.env_checker import check_env


# Single module loading utils
OPTIONS_LLM_RATER = [
    "2Shot_system_our",
    "1Shot_system_our",
    "0Shot_system_our",
    "2Shot_system_default",
    "1Shot_system_default",
    "0Shot_system_default",
    "0Shot_system_our_1_10",
    "1Shot_system_our_1_10",
    "2Shot_system_our_1_10",
    "2Shot_system_our_one_ten",
    "1Shot_system_our_one_ten",
    "2Shot_invert_system_default",
    "2Shot_invert_system_our",
    "1Shot_invert_system_our",
    "1Shot_invert_system_default",
]
OPTIONS_ITEMS_RETRIEVAL = ["last_3", "most_similar_3", "none", "simple_3"]
OPTIONS_REWARD_PERTURBATOR = ["none", "gaussian", "greedy"]
OPTIONS_USER_DATASET = ["detailed", "basic", "sampled_genres"]
OPTIONS_REWARD_SHAPING = [
    "identity",
    "exp_decay_time",
    "random_watch",
    "same_film_terminate",
]


def get_llm_rater(name, llm, history=True):
    CURRENT_MOVIE_FEATURES_LIST = [
        "title",
        "overview",
        "genres",
        "actors",
        "vote_average",
    ]
    if name == "2Shot_system_our":
        return ThirdPersonDescriptive09_2Shot_OurSys(
            llm,
            current_items_features_list=CURRENT_MOVIE_FEATURES_LIST,
            previous_items_features_list=["title", "rating"] if history else [],
        )
    elif name == "2Shot_invert_system_our":
        return ThirdPersonDescriptive09_2Shot_OurSys(
            llm,
            current_items_features_list=CURRENT_MOVIE_FEATURES_LIST,
            previous_items_features_list=["title", "rating"] if history else [],
            switch_order=True,
        )
    elif name == "1Shot_system_our":
        return ThirdPersonDescriptive09_1Shot_OurSys(
            llm,
            current_items_features_list=CURRENT_MOVIE_FEATURES_LIST,
            previous_items_features_list=["title", "rating"] if history else [],
        )
    elif name == "1Shot_invert_system_our":
        return ThirdPersonDescriptive09_1Shot_OurSys(
            llm,
            current_items_features_list=CURRENT_MOVIE_FEATURES_LIST,
            previous_items_features_list=["title", "rating"] if history else [],
            switch_user=True,
        )
    elif name == "0Shot_system_our":
        return ThirdPersonDescriptive09_OurSys(
            llm,
            current_items_features_list=CURRENT_MOVIE_FEATURES_LIST,
            previous_items_features_list=["title", "rating"] if history else [],
        )
    elif name == "2Shot_system_default":
        return ThirdPersonDescriptive09_2Shot(
            llm,
            current_items_features_list=CURRENT_MOVIE_FEATURES_LIST,
            previous_items_features_list=["title", "rating"] if history else [],
        )
    elif name == "2Shot_invert_system_default":
        return ThirdPersonDescriptive09_2Shot(
            llm,
            current_items_features_list=CURRENT_MOVIE_FEATURES_LIST,
            previous_items_features_list=["title", "rating"] if history else [],
            switch_order=True,
        )
    elif name == "1Shot_system_default":
        return ThirdPersonDescriptive09_1Shot(
            llm,
            current_items_features_list=CURRENT_MOVIE_FEATURES_LIST,
            previous_items_features_list=["title", "rating"] if history else [],
        )
    elif name == "1Shot_invert_system_default":
        return ThirdPersonDescriptive09_1Shot(
            llm,
            current_items_features_list=CURRENT_MOVIE_FEATURES_LIST,
            previous_items_features_list=["title", "rating"] if history else [],
            switch_user=True,
        )
    elif name == "0Shot_system_default":
        return ThirdPersonDescriptive09(
            llm,
            current_items_features_list=CURRENT_MOVIE_FEATURES_LIST,
            previous_items_features_list=["title", "rating"] if history else [],
        )
    elif name == "2Shot_system_our_1_10":
        return ThirdPersonDescriptive110_2Shot_OurSys(
            llm,
            current_items_features_list=CURRENT_MOVIE_FEATURES_LIST,
            previous_items_features_list=["title", "rating"] if history else [],
        )
    elif name == "1Shot_system_our_1_10":
        return ThirdPersonDescriptive110_1Shot_OurSys(
            llm,
            current_items_features_list=CURRENT_MOVIE_FEATURES_LIST,
            previous_items_features_list=["title", "rating"] if history else [],
        )
    elif name == "0Shot_system_our_1_10":
        return ThirdPersonDescriptive110_OurSys(
            llm,
            current_items_features_list=CURRENT_MOVIE_FEATURES_LIST,
            previous_items_features_list=["title", "rating"] if history else [],
        )
    elif name == "2Shot_system_our_one_ten":
        return ThirdPersonDescriptiveOneTen_2Shot_OurSys(
            llm,
            current_items_features_list=CURRENT_MOVIE_FEATURES_LIST,
            previous_items_features_list=["title", "rating"] if history else [],
        )
    elif name == "1Shot_system_our_one_ten":
        return ThirdPersonDescriptiveOneTen_1Shot_OurSys(
            llm,
            current_items_features_list=CURRENT_MOVIE_FEATURES_LIST,
            previous_items_features_list=["title", "rating"] if history else [],
        )
    else:
        raise ValueError(f"Unknown LLM rater {name}")


def get_items_retrieval(name):
    if name == "last_3":
        return TimeItemsRetrieval(3)
    elif name == "most_similar_3":
        return SentenceSimilarityItemsRetrieval(3, "overview_embedding")
    elif name == "simple_3":
        return SimpleMoviesRetrieval(3)
    elif name == "none":
        return TimeItemsRetrieval(0)
    else:
        raise ValueError(f"Unknown item retrieval {name}")


def get_reward_perturbator(name, seed):
    if name == "none":
        return NoPerturbator(seed=seed, stepsize=1.0)
    elif name == "gaussian":
        return GaussianPerturbator(seed=seed, stepsize=1.0)
    elif name == "greedy":
        return GreedyPerturbator(seed=seed, stepsize=1.0)


def get_user_dataset(name):
    base_dir = os.path.join(
        os.path.dirname(__file__),
        "./users_generation/datasets/",
    )

    if name == "detailed":
        return UsersCSVLoader("user_features_hard_600", base_dir)
    elif name == "basic":
        return UsersCSVLoader("user_features_600", base_dir)
    elif name == "sampled_genres":
        return UsersCSVLoader("user_features_sampled_genres_600", base_dir)
    else:
        raise ValueError(f"Unknown user dataset {name}")


def get_reward_shaping(name, seed):
    if name == "identity":
        return IdentityRewardShaping()
    elif name == "exp_decay_time":
        return RewardReshapingExpDecayTime(q=0.1, seed=seed, stepsize=1.0)
    elif name == "random_watch":
        return RewardReshapingRandomWatch(q=0.1, seed=seed, stepsize=1.0)
    elif name == "same_film_terminate":
        return RewardReshapingTerminateIfSeen(
            q=0.1,
            seed=seed,
            stepsize=1.0,
        )
    else:
        raise ValueError(f"Unknown reward shaping {name}")


def get_base_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llm-model",
        type=str,
        default="TheBloke/Llama-2-7b-Chat-GPTQ",
        choices=LLM.SUPPORTED_MODELS,
    )

    parser.add_argument(
        "--llm-rater",
        type=str,
        default="2Shot_system_our",
        choices=OPTIONS_LLM_RATER,
    )
    parser.add_argument(
        "--items-retrieval",
        type=str,
        default="last_3",
        choices=OPTIONS_ITEMS_RETRIEVAL,
    )

    parser.add_argument(
        "--user-dataset",
        type=str,
        default="sampled_genres",
        choices=OPTIONS_USER_DATASET,
    )
    parser.add_argument(
        "--film-dataset",
        type=str,
        default="movielens_latest-small",
        choices=["movielens_latest-small"],
    )
    parser.add_argument(
        "--perturbator",
        type=str,
        default="none",
        choices=OPTIONS_REWARD_PERTURBATOR,
    )
    parser.add_argument(
        "--reward-shaping",
        type=str,
        default="exp_decay_time",
        choices=OPTIONS_REWARD_SHAPING,
    )

    parser.add_argument("--seed", type=int, default=42)
    return parser


def get_enviroment_from_args(
    llm, args, seed=None, render_mode=None, render_path=None, eval_mode=False
):
    """Returns the environment with the configuration specified in args."""
    if seed is None:
        seed = args.seed
    env = Simulatio4RecSys(
        render_mode=render_mode,
        render_path=render_path,
        items_loader=MoviesLoader(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "./datasets/",
                args.film_dataset + ".json",
            )
        ),
        users_loader=get_user_dataset(args.user_dataset),
        items_selector=GreedySelector(seed),
        reward_perturbator=get_reward_perturbator(args.perturbator, seed),
        items_retrieval=get_items_retrieval(args.items_retrieval),
        llm_rater=get_llm_rater(
            args.llm_rater, llm, history=args.items_retrieval != "none"
        ),
        reward_shaping=get_reward_shaping(args.reward_shaping, seed),
        evaluation=eval_mode,
    )
    env.reset(seed=seed)
    return env
