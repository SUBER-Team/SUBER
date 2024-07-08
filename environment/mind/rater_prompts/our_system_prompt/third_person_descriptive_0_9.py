import os
from typing import List
from environment.LLM import LLMRater
from environment.LLM.llm import LLM
import re
import pandas as pd
from environment.memory import UserNewsInteraction
from environment.mind import News, NewsLoader
from environment.users import User


import logging
from algorithms.logging_config  import get_logger

logger = get_logger("suber_logger")


class ThirdPersonDescriptive09_OurSys(LLMRater):
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
        

        self.system_prompt = (
            "You are a highly sophisticated news rating assistant, equipped with an"
            " advanced understanding of human behavior. Your mission is to deliver"
            " personalized news recommendations by carefully considering the unique"
            " characteristics, tastes, and past seen news of each individual. When"
            " presented with information about a specific news article, you will diligently"
            " analyze it, primary catagories, persons, places, and average rating. Using this"
            " comprehensive understanding, your role is to provide thoughtful and"
            " accurate ratings for news on a scale of 0 to 9, ensuring they resonate"
            " with the person's preferences and cinematic inclinations. Remain"
            " impartial and refrain from introducing any biases in your predictions."
            " You are an impartial and reliable source of news rating predictions for"
            " the given individual and film descriptions."
        )

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
        news: News,
        num_interacted: int,
        interactions: List[UserNewsInteraction],
        retrieved_items: List[News],
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
                f'\n - "{m.title}" ({int(self.adjust_rating_in(i.rating))}), '
            )
        if len(retrieved_items) > 0:
            item_interaction = item_interaction[:-2]  # remove last comma
        
        news_cats_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../../datasets/news_cats.csv",
    )
        news_catagories_list = pd.read_csv(news_cats_path, header=None)[0].tolist()
        news_catagories_list_string = ""
        for g in news_catagories_list:
            
            news_catagories_list_string += f"-{g}\n"



        if len(news.abstract) > 0:
            overview = news.abstract[0].lower() + news.abstract[1:]

        else:
            overview = ""

        name = user.name.split(" ")[0]
        # NOTE: this is a hack to make sure that the name is not the same as the 2 possible names used in the few-shot prompts
        name = self.adjust_text_in(name, do_rename)

        prompt = (
            f"{name} is a {user.age} years old {gender},"
            f" {pronoun} is {self.adjust_text_in(user.description, do_rename)}\n"
            + (
                f"\n{name} has previously read the following news articles (in"
                " parentheses are the ratings he gave on a scale of 0 to 9):"
                f" {item_interaction}.\n"
                if len(retrieved_items) > 0
                and len(self.previous_items_features_list) > 0
                else ""
            )
            + f'\nConsider the news article entitled "{news.title}".'
            f" It is described as follows: {overview}\n"
            + (
                f'\nThe news article is categorized as "{news.category}" with a subcategory of "{news.subcategory}". '
                #f"It contains the following named entities:\n ent1, ent2\n\n" # TODO make code ot extract entities from news
            ) # TODO Should we reword this so it's categories and subcategories? 
         
            + (
                f'On average, the new article as a click-through rate of {news.click_through_rate} and ' # TODO Need to work on the behaviors to see and count impressions and such
                f"the news article has been read {news.read_frequency} times.\n\n"

            )
            + f'{name} has read the news article, "{news.title}", for'
            f" {self.number_to_rank(num_interacted+1)} times.\n"
            + f"\nWhat can you conclude about {name}'s rating for the news article"
            f' "{news.title}" on a scale of 0 to 9, where 0 represents a low rating'
            " and 9 represents a high rating, based on available information and"
            " logical reasoning?"
        )

        initial_assistant = (
            f"Based on {name}'s preferences and tastes, I conclude that {pronoun} will"
            " assign a rating of "
        )
        #logger.info("PROMPT: {}".format(prompt))
        #logger.info("initial_assistant:  {}".format(initial_assistant))
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
