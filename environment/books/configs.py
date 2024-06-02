import argparse
import os

from environment import LLM
from environment.books.books_loader import BooksLoader
from environment.books.books_retrieval import SimpleBookRetrieval
from ..env import Simulatio4RecSys
from ..users import UsersCSVLoader
from ..items_retrieval import (
    SentenceSimilarityItemsRetrieval,
    TimeItemsRetrieval,
)
from ..items_selection import GreedySelector
from ..reward_perturbator import GaussianPerturbator, GreedyPerturbator, NoPerturbator
from .rater_prompts.our_system_prompt import (
    ThirdPersonDescriptive15_2Shot_OurSys,
    ThirdPersonDescriptive15_OurSys,
    ThirdPersonDescriptive15_1Shot_OurSys,
)
from environment.reward_shaping import (
    RewardReshapingExpDecayTime,
    RewardReshapingRandomWatch,
    IdentityRewardShaping,
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
    # "2Shot_system_our_one_five",
    # "1Shot_system_our_one_five",
]
OPTIONS_ITEMS_RETRIEVAL = ["last_3", "most_similar_3", "none", "simple_3"]
OPTIONS_REWARD_PERTURBATOR = ["none", "gaussian", "greedy"]
OPTIONS_USER_DATASET = ["detailed", "sampled"]
OPTIONS_REWARD_SHAPING = ["identity", "exp_decay_time", "random_watch"]


def get_llm_rater(name, llm, history=True):
    CURRENT_MOVIE_FEATURES_LIST = [
        "title",
        "description",
        "categories",
        "authors",
        "vote_average",
    ]
    if name == "2Shot_system_our":
        return ThirdPersonDescriptive15_2Shot_OurSys(
            llm,
            current_items_features_list=CURRENT_MOVIE_FEATURES_LIST,
            previous_items_features_list=["title", "rating"] if history else [],
            system_prompt="our_system_prompt",
        )
    elif name == "1Shot_system_our":
        return ThirdPersonDescriptive15_1Shot_OurSys(
            llm,
            current_items_features_list=CURRENT_MOVIE_FEATURES_LIST,
            previous_items_features_list=["title", "rating"] if history else [],
            system_prompt="our_system_prompt",
        )
    elif name == "0Shot_system_our":
        return ThirdPersonDescriptive15_OurSys(
            llm,
            current_items_features_list=CURRENT_MOVIE_FEATURES_LIST,
            previous_items_features_list=["title", "rating"] if history else [],
            system_prompt="our_system_prompt",
        )
    elif name == "2Shot_system_default":
        return ThirdPersonDescriptive15_2Shot_OurSys(
            llm,
            current_items_features_list=CURRENT_MOVIE_FEATURES_LIST,
            previous_items_features_list=["title", "rating"] if history else [],
            system_prompt=None,
        )
    elif name == "1Shot_system_default":
        return ThirdPersonDescriptive15_1Shot_OurSys(
            llm,
            current_items_features_list=CURRENT_MOVIE_FEATURES_LIST,
            previous_items_features_list=["title", "rating"] if history else [],
            system_prompt=None,
        )
    elif name == "0Shot_system_default":
        return ThirdPersonDescriptive15_OurSys(
            llm,
            current_items_features_list=CURRENT_MOVIE_FEATURES_LIST,
            previous_items_features_list=["title", "rating"] if history else [],
            system_prompt=None,
        )
    else:
        raise ValueError(f"Unknown LLM rater {name}")


def get_items_retrieval(name):
    if name == "last_3":
        return TimeItemsRetrieval(3)
    elif name == "most_similar_3":
        return SentenceSimilarityItemsRetrieval(3, "description_embedding")
    elif name == "simple_3":
        return SimpleBookRetrieval(3)
    elif name == "none":
        return TimeItemsRetrieval(0)
    else:
        raise ValueError(f"Unknown item retrieval {name}")


def get_reward_perturbator(name, seed):
    if name == "none":
        return NoPerturbator(seed=seed, stepsize=1.0, min_rating=1.0, max_rating=5.0)
    elif name == "gaussian":
        return GaussianPerturbator(
            seed=seed, stepsize=1.0, min_rating=1.0, max_rating=5.0
        )
    elif name == "greedy":
        return GreedyPerturbator(
            seed=seed, stepsize=1.0, min_rating=1.0, max_rating=5.0
        )


def get_user_dataset(name):
    base_dir = os.path.join(
        os.path.dirname(__file__),
        "./users_generation/datasets/",
    )
    if name == "detailed":
        return UsersCSVLoader("users_600", base_dir)
    elif name == "sampled":
        return UsersCSVLoader("user_features_sampled_categories_600", base_dir)
    else:
        raise ValueError(f"Unknown user dataset {name}")


def get_reward_shaping(name, seed):
    if name == "identity":
        return IdentityRewardShaping(min_rating=1.0, max_rating=5.0)
    elif name == "exp_decay_time":
        return RewardReshapingExpDecayTime(
            q=0.1, seed=seed, stepsize=1.0, min_rating=1.0, max_rating=5.0
        )
    elif name == "random_watch":
        return RewardReshapingRandomWatch(
            q=0.1, seed=seed, stepsize=1.0, min_rating=1.0, max_rating=5.0
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
        default="most_similar_3",
        choices=OPTIONS_ITEMS_RETRIEVAL,
    )

    parser.add_argument(
        "--user-dataset",
        type=str,
        default="detailed",
        choices=OPTIONS_USER_DATASET,
    )
    parser.add_argument(
        "--book-dataset",
        type=str,
        default="books_amazon/postprocessed_books",
        choices=["books_amazon/postprocessed_books"],
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


def get_enviroment_from_args(llm, args, seed=None, render_mode=None):
    """Returns the environment with the configuration specified in args."""
    if seed is None:
        seed = args.seed
    env = Simulatio4RecSys(
        render_mode=None,
        items_loader=BooksLoader(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "./datasets/",
                args.book_dataset + "_embeddings" + ".csv",
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
    )
    env.reset(seed=seed)
    check_env(env)
    return env
