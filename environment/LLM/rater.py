from abc import ABC, abstractmethod
from typing import List, Tuple
from environment.LLM.llm import LLM

from environment.memory import UserMovieInteraction
from environment.movies.movie import Movie
from environment.users import User
import numpy as np


class LLMRater(ABC):
    """
    Abstract class that defines the interface for the prompting system.
    The prompting system is used to generate the prompt for the LLM.
    The prompt is generated based on the user, the item and the previous interactions.
    Numbers in user description, and rating can by adjusted using functions adjust_text_in, adjust_rating_in, adjust_rating_out.
    """

    def __init__(
        self,
        llm: LLM,
        current_items_features_list,
        previous_items_features_list,
        llm_render=False,
        llm_query_explanation=False,
    ):
        self.llm = llm
        self.llm_render = llm_render
        self.llm_query_explanation = llm_query_explanation
        self.current_items_features_list = current_items_features_list
        self.previous_items_features_list = previous_items_features_list

        self.system_prompt = None
        self.request_scale = "0-9"
        self.random_rating = False
        self.rnd = np.random.RandomState(42)

    @abstractmethod
    def adjust_rating_in(self, rating):
        pass

    @abstractmethod
    def adjust_rating_out(self, rating):
        pass

    @abstractmethod
    def adjust_text_in(self, text):
        pass

    def query(
        self,
        user: User,
        item: Movie,
        num_interacted: int,
        interactions: List[UserMovieInteraction],
        retrieved_items: List[Movie],
    ) -> Tuple[int, str, str]:
        """
        Queries the LLM for the rating of the item, using the library guidance.

        Args:
            user (User): the user
            item (Movie): the item
            num_interacted (int): the number of times the item has been watched
            interactions (list of UserMovieInteraction): the previous interactions
            retrieved_items (list of Movie): the retrieved items

        Returns:
            Tuple[int, str, str]: the rating, the explanation of the LLM and html of LLM interaction (if llm_query_explanation is True)

        """
        few_shot_prompts = self._get_few_shot_prompts()
        prompt = self._get_prompt(
            user, item, num_interacted, interactions, retrieved_items
        )

        if self.request_scale == "0-9":
            _, out = self.llm.request_rating_0_9(
                self.system_prompt, few_shot_prompts + prompt
            )
            try:
                rating = self.adjust_rating_out(float(out))
            except Exception:
                rating = float(0)
            if self.random_rating:
                rating = self.rnd.randint(1, 11)
        elif self.request_scale == "1-10":
            _, out = self.llm.request_rating_1_10(
                self.system_prompt, few_shot_prompts + prompt
            )
            try:
                rating = self.adjust_rating_out(float(out))
            except Exception:
                rating = float(0)
            if self.random_rating:
                rating = self.rnd.randint(1, 11)
        elif self.request_scale == "1-5":
            _, out = self.llm.request_rating_1_5(
                self.system_prompt, few_shot_prompts + prompt
            )
            try:
                rating = self.adjust_rating_out(float(out))
            except Exception:
                rating = float(0)
            if self.random_rating:
                rating = self.rnd.randint(1, 6)
        else:
            _, out = self.llm.request_rating_text(
                self.system_prompt, few_shot_prompts + prompt
            )
            try:
                m = {
                    "one": 1,
                    "two": 2,
                    "three": 3,
                    "four": 4,
                    "five": 5,
                    "six": 6,
                    "seven": 7,
                    "eight": 8,
                    "nine": 9,
                    "ten": 10,
                }  # map 1 to 10
                rating = float(m[out])
            except ValueError:
                rating = float("nan")
            if self.random_rating:
                rating = self.rnd.randint(1, 11)

        if self.llm_query_explanation:
            prompt_explanation = self._get_prompt_explanation(prompt, rating)
            prompt_txt, explanation = self.llm.request_explanation(
                self.system_prompt, few_shot_prompts + prompt_explanation
            )

            if self.llm_render:
                print("-" * 80)
                print(prompt_txt + explanation)
            out = (
                "Question:\n"
                + prompt_explanation[0]["content"]
                + "\nAnswer:\n"
                + prompt_explanation[1]["content"]
                + explanation
            )
            return rating, explanation, out
        else:
            return rating, "", ""

    def number_to_rank(self, number):
        """
        Converts a number to a string with the corresponding suffix (e.g. 1 -> 1st, 2 -> 2nd, 3 -> 3rd, ...)

        Args:
            number (int): the number to convert

        Returns:
            str: the number with the corresponding suffix
        """
        if number % 100 in (11, 12, 13):
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
        return str(number) + suffix
