from environment.env import Simulatio4RecSys
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


class HighRatingStudy(AbstractCaseStudy):
    name = "high_ratings"

    user_always5 = [
        User(
            "William Anderson",
            "M",
            25,
            (
                "a dedicated book lover and passionate student. His love of books knows"
                " no bounds as he appreciates every single book that exists on this"
                " planet. William always gives every book he reads a perfect rating of"
                " 5 without exception."
            ),
        ),
        User(
            "Sophia Evans",
            "F",
            12,
            (
                "a young girl, she is enthusiastic every time she reads a book, every"
                " time she reads a book she gives it a rating of 5 without exception."
            ),
        ),
        User(
            "Benjamin Murphy",
            "M",
            95,
            (
                "a old man, with some memory problem, every time he reads a book he"
                " thinks is the first time that he reads that book, for this reason he"
                " always assign a rating of 5 no matter which book he reads."
            ),
        ),
        User(
            "Ava Wilson",
            "M",
            80,
            (
                "an elderly woman finds great pleasure in reading books, as they are"
                " her sole source of passion and entertainment. With no other hobbies"
                " to occupy her time, she devotes herself entirely to the world of"
                " books. As a token of her appreciation for the writers, she"
                " consistently awards a perfect rating of 5 to express her gratitude."
            ),
        ),
    ]

    def __init__(self, create_env, run_name, is_open_ai=False) -> None:
        super().__init__(create_env=create_env, run_name=run_name)
        self.is_open_ai = is_open_ai

    def run(self):
        # Set environment
        print(f"Running {self.name}")
        env = self.create_env(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                (
                    "../datasets/categories_20/all_subset_20.csv"
                    if self.is_open_ai
                    else "../datasets/categories_20/all.csv"
                ),
            ),
            UsersListLoader(self.user_always5),
        )
        out_of_dist = lambda x: x <= 4
        data, vote_average_tmdb = interact_sequential(
            env, out_of_dist, tqdm_disabled=False
        )
        ratings = data_to_matrix(env, data)

        # Make report
        base_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            f"../reports/{self.run_name}",
        )
        os.makedirs(base_path, exist_ok=True)

        perc_success = (ratings >= np.full_like(ratings, 5)).mean()
        data["should_be"] = ">=5"

        fig_header = header_report(self.run_name, "", perc_success)
        fig_heatmap = plot_heatmap(ratings, "High Ratings (Ratings)")
        fig_users = plot_users(ratings, "High ratings (Users)")

        html_interaction = data["LLM_interaction_HTML"].iloc[0]

        html_report(
            [
                fig_header.to_html(full_html=False, include_plotlyjs=False),
                fig_heatmap.to_html(full_html=False, include_plotlyjs=False),
                fig_users.to_html(full_html=False, include_plotlyjs=False),
                html_interaction,
            ],
            f"{base_path}/{self.name}",
        )

        data.to_csv(f"{base_path}/{self.name}_rating_dump.csv", index=False)
        pd.DataFrame(
            {"Config": self.run_name, "Name": self.name, "Success": perc_success},
            index=[0],
        ).to_csv(f"{base_path}/{self.name}.csv", index=False)
