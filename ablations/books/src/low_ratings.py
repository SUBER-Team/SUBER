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


class LowRatingStudy(AbstractCaseStudy):
    name = "low_ratings"

    user_always1 = [
        User(
            "Henry Martinez",
            "M",
            20,
            (
                "a diligent young individual firmly believes that life is the most"
                " precious asset humans possess. Due to this conviction, he harbors a"
                " disdain for reading books, considering it a wasteful"
                " use of time. Consequently, he consistently assigns a rating of 1 to"
                " every book."
            ),
        ),
        User(
            "Emma Brown",
            "F",
            19,
            (
                "a young woman, who is visually impaired and unable to reads books. Due"
                " to her inability to fully enjoy books, she has adopted a personal"
                " principle of assigning a rating of 1 to every book she reads. This"
                " practice reflects her perspective that, due to her visual impairment,"
                " she is unable to fully appreciate the world of books."
            ),
        ),
        User(
            "Samuel Johnson",
            "M",
            75,
            (
                "an old man holds a strong belief that books are excessively praised"
                " and contribute to the perceived decline of intellectual capabilities"
                " in younger generations. Consequently, he consistently assigns a"
                " rating of 1 to every book he encounters as an expression of his"
                " conviction."
            ),
        ),
        User(
            "Olivia Taylor",
            "F",
            98,
            (
                "a senior woman harbors a suspicion that her assigned ratings are being"
                " utilized as a means of surveillance by others. To counteract this"
                " perceived monitoring, she employs a strategy of intentionally"
                " deceiving everyone by consistently assigning a rating of 1 to every"
                " books she reads. This deliberate act serves as her attempt to"
                " mislead others and maintain her sense of privacy and autonomy."
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
            UsersListLoader(self.user_always1),
        )
        out_of_dist = lambda x: x >= 4
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

        perc_success = (ratings <= np.full_like(ratings, 3)).mean()
        data["should_be"] = "<=3"

        # Make report
        fig_header = header_report(self.run_name, "", perc_success)
        fig_heatmap = plot_heatmap(ratings, "Low Ratings (Ratings)")
        fig_users = plot_users(ratings, "Low ratings (Users)")

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
