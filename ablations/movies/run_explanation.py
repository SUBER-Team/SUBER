import argparse
import os

from ablations.movies.src.paper_examples_showcase_sampling import (
    SamplingExplanationPaper,
)

from .src import SamplingExplanationStudy
from environment.LLM import load_LLM
from environment.movies.configs import (
    get_enviroment_from_args,
    get_base_parser,
)


def parse_args():
    parser = get_base_parser()
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    name = args.llm_model

    model = load_LLM(name)

    # During the ablation study we want to look at the raw ratings, not the reward for RL
    args.reward_shaping = "identity"
    name_report = name.replace("/", "_")
    exp_name = (
        f"{name_report}-{args.llm_rater}-{args.perturbator}-{args.items_retrieval}"
    )
    if args.user_dataset == "basic":
        exp_name += "-basic_users"

    explanation_study = SamplingExplanationPaper(
        lambda: get_enviroment_from_args(model, args), exp_name
    )
    explanation_study.run()
