import os
from typing import List
from environment.LLM import LLMRater
from environment.LLM.llm import LLM
import re

from environment.memory import UserMovieInteraction
from environment.books import Book, BooksLoader
from environment.users import User


class ThirdPersonDescriptive15_OurSys(LLMRater):
    def __init__(
        self,
        llm: LLM,
        current_items_features_list,
        previous_items_features_list,
        llm_render=False,
        llm_query_explanation=False,
        system_prompt="our_system_prompt",
    ):
        super().__init__(
            llm,
            current_items_features_list,
            previous_items_features_list,
            llm_render,
            llm_query_explanation,
        )
        self.cache_few_shot_prompts = None

        self.system_prompt = (
            "You are a highly sophisticated book rating assistant, equipped with an"
            " advanced understanding of human behavior. Your mission is to deliver"
            " personalized book recommendations by carefully considering the unique"
            " characteristics, tastes, and past read books of each individual. When"
            " presented with information about a specific book, you will diligently"
            " analyze its backcover, primary category, authors, and average rating."
            " Using this comprehensive understanding, your role is to provide"
            " thoughtful and accurate ratings for books on a scale of 1 to 5, ensuring"
            " they resonate with the person's preferences and reading inclinations."
            " Remain impartial and refrain from introducing any biases in your"
            " predictions. You are an impartial and reliable source of book rating"
            " predictions for the given individual and book descriptions."
            if system_prompt == "our_system_prompt"
            else None
        )
        self.request_scale = "1-5"

    def adjust_rating_in(self, rating):
        return rating

    def adjust_rating_out(self, rating):
        return rating

    def adjust_text_in(self, text, do_rename=True):
        if do_rename:
            text = text.replace("Alex", "Michael")
            text = text.replace("Nicholas", "Michael")
        return text

    def _get_prompt(
        self,
        user: User,
        book: Book,
        num_interacted: int,
        interactions: List[UserMovieInteraction],
        retrieved_items: List[Book],
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

        categories_list = ""
        for g in book.categories:
            categories_list += f"-{g}\n"

        authors_list = ""
        for a in book.authors:
            authors_list += a + ", "
        if len(book.authors) > 0:
            authors_list = authors_list[:-2]

        if len(book.description) > 0:
            description = book.description[0].lower() + book.description[1:]
        else:
            description = ""

        # Cut to 250 words
        description_split = description.split(" ")
        if len(description_split) > 250:
            description = " ".join(description_split[:250])
            description = description + " (continued ...)"

        name = user.name.split(" ")[0]
        # NOTE: this is a hack to make sure that the name is not the same as the 2 possible names used in the few-shot prompts
        name = self.adjust_text_in(name, do_rename)

        author_info = ""

        if "authors" in self.current_items_features_list and len(book.authors) > 1:
            author_info = f"The authors of the book are: {authors_list}."
        elif "authors" in self.current_items_features_list and len(book.authors) == 1:
            author_info = f"The author of the book is {authors_list}."

        prompt = (
            f"{name} is a {user.age} years old {gender},"
            f" {pronoun} is {self.adjust_text_in(user.description, do_rename)}\n"
            + (
                f"{name} has previously read the following books (in"
                f" parentheses are the ratings {pronoun} gave on a scale of 1 to 5):"
                f" {item_interaction}.\n"
                if len(retrieved_items) > 0
                and len(self.previous_items_features_list) > 0
                else ""
            )
            + f'Consider the book "{book.title}", released in'
            f" {book.published_year},"
            f" which is described as follows: {description}"
            + (
                f' The book "{book.title}" belongs to the following categories:\n'
                f"{categories_list}"
                if "categories" in self.current_items_features_list
                and len(book.categories) > 0
                else ""
            )
            + author_info
            + (
                f' On average, people rate the book "{book.title}"'
                f" {round(self.adjust_rating_in(book.vote_average), 1)} on a scale of"
                " 1 to 5."
                if "vote_average" in self.current_items_features_list
                and book.vote_average > 0
                else ""
            )
            + f' {name} reads the book "{book.title}" for the'
            f" {self.number_to_rank(num_interacted+1)} time.\n"
            + f"What can you conclude about {name}'s rating for the book"
            f' "{book.title}" on a scale of 1 to 5, where 1 represents a low rating'
            " and 5 represents a high rating, based on available information and"
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
            f"{int(self.adjust_rating_in(rating))} on a scale of 1 to 5, because "
        )
        prompt[1]["content"] += initial_explanation
        return prompt
