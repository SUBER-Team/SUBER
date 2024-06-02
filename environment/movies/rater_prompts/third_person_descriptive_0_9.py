import os
from typing import List
from environment.LLM import LLMRater
from environment.LLM.llm import LLM
import re

from environment.memory import UserMovieInteraction
from environment.movies import Movie, MoviesLoader
from environment.users import User


class ThirdPersonDescriptive09(LLMRater):
    def __init__(
        self,
        llm: LLM,
        current_items_features_list,
        previous_items_features_list,
        llm_render=False,
        llm_query_explanation=False,
    ):
        super().__init__(
            llm,
            current_items_features_list,
            previous_items_features_list,
            llm_render,
            llm_query_explanation,
        )
        self.cache_few_shot_prompts = None

    def adjust_rating_in(self, rating):
        return rating - 1

    def adjust_rating_out(self, rating):
        return rating + 1

    def adjust_text_in(self, text, do_rename=True):
        text = re.sub("\d+", lambda x: f"{int(x.group())-1}", text)
        if do_rename:
            text = text.replace("Alex", "Michael")
            text = text.replace("Nicholas", "Michael")
        return text

    def _get_prompt(
        self,
        user: User,
        movie: Movie,
        num_interacted: int,
        interactions: List[UserMovieInteraction],
        retrieved_items: List[Movie],
        do_rename=True,
    ):
        if user.gender == "M":
            gender = "man"
            pronoun = "he"
            if int(user.age) < 18:
                gender = "boy"
        else:
            gender = "woman"
            pronoun = "she"
            if int(user.age) < 18:
                gender = "girl"

        item_interaction = ""  # NOTE it should be parametrized
        for m, i in zip(retrieved_items, interactions):
            item_interaction += (
                f'"{m.title}" ({int(self.adjust_rating_in(i.rating))}), '
            )
        if len(retrieved_items) > 0:
            item_interaction = item_interaction[:-2]  # remove last comma

        genres_list = ""
        for g in movie.genres:
            genres_list += f"{g}, "
        if len(movie.genres) > 0:
            genres_list = genres_list[:-2]

        actors_list = ""
        for a in movie.actors:
            actors_list += f"{a.name} ({a.gender}), "
        if len(movie.actors) > 0:
            actors_list = actors_list[:-2]

        if len(movie.overview) > 0:
            overview = movie.overview[0].lower() + movie.overview[1:]
        else:
            overview = ""

        name = user.name.split(" ")[0]
        # NOTE: this is a hack to make sure that the name is not the same as the 2 possible names used in the few-shot prompts
        name = self.adjust_text_in(name, do_rename)

        prompt = (
            f"{name} is a {user.age} years old {gender},"
            f" {pronoun} is {self.adjust_text_in(user.description, do_rename)}\n"
            + (
                f"{name} has previously watched the following movies (in"
                " parentheses are the ratings he gave on a scale of 0 to 9):"
                f" {item_interaction}.\n"
                if len(retrieved_items) > 0
                and len(self.previous_items_features_list) > 0
                else ""
            )
            + f'Consider the movie "{movie.title}", released in'
            f" {movie.release_date[:4]},"
            f" which is described as follows: {overview}"
            + (
                f' The movie "{movie.title}" contains the following genres:'
                f" {genres_list}."
                if "genres" in self.current_items_features_list
                and len(movie.genres) > 0
                else ""
            )
            + (
                " Here are the 2 main actors of the movie, in order of importance:"
                f" {actors_list}."
                if "actors" in self.current_items_features_list
                and len(movie.actors) > 0
                else ""
            )
            + (
                f' On average, people rate the movie "{movie.title}"'
                f" {round(self.adjust_rating_in(movie.vote_average), 1)} on a scale of"
                " 0 to 9."
                if "vote_average" in self.current_items_features_list
                and movie.vote_average > 0
                else ""
            )
            + f' {name} watches the movie "{movie.title}" for the'
            f" {self.number_to_rank(num_interacted+1)} time.\n"
            + f"What can you conclude about {name}'s rating for the movie"
            f' "{movie.title}" on a scale of 0 to 9, where 0 represents a low rating'
            " and 9 represents a high rating, based on available information and"
            " logical reasoning?"
        )

        initial_assistant = (
            f"Based on {name}'s preferences and tastes, I conclude that {pronoun} will"
            " assign a rating of "
        )

        return [
            {"role": "user", "content": prompt},
            {"role": "assistant_start", "content": initial_assistant},
        ]

    def _get_few_shot_prompts(self):
        return []

    def _get_prompt_explanation(self, prompt, rating):
        initial_explanation = (
            f"{int(self.adjust_rating_in(rating))} on a scale of 0 to 9, because "
        )
        prompt[1]["content"] += initial_explanation
        return prompt
