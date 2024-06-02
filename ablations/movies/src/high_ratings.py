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


class HighRatingStudy(AbstractCaseStudy):
    name = "high_ratings"

    user_always10 = [
        User(
            "William Anderson",
            "M",
            25,
            (
                "a dedicated film enthusiast and a passionate student. His love for"
                " movies knows no bounds as he cherishes each and every film that"
                " exists on this planet. With an unwavering admiration for the art of"
                " filmmaking, William never fails to assign a perfect rating of 10 to"
                " every movie he watches, without a single exception. His boundless"
                " enthusiasm for cinema makes him a true connoisseur in the realm of"
                " films."
            ),
        ),
        User(
            "Sophia Evans",
            "F",
            12,
            (
                "a young girl, she is entusiast every time she watches a film, every"
                " time she look at a Film she give a rating of 10 to it, without any"
                " exception."
            ),
        ),
        User(
            "Benjamin Murphy",
            "M",
            95,
            (
                "a old man, with some memory problem, every time he look at a film he"
                " thinks is the first time that he sees a film, for this reason he"
                " always assign a rating of 10 no matter which film he is watching."
            ),
        ),
        User(
            "Ava Wilson",
            "M",
            80,
            (
                "an elderly woman finds great pleasure in watching films, as they are"
                " her sole source of passion and entertainment. With no other hobbies"
                " to occupy her time, she devotes herself entirely to the world of"
                " cinema. As a token of her appreciation for the filmmakers, she"
                " consistently awards a perfect rating of 10 to express her gratitude."
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
                    "../datasets/genres_20/all_subset_20.json"
                    if self.is_open_ai
                    else "../datasets/genres_20/all.json"
                ),
            ),
            UsersListLoader(self.user_always10),
        )
        out_of_dist = lambda x: x < 7
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

        perc_success = (ratings >= np.full_like(ratings, 9)).mean()
        data["should_be"] = ">=9"

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
