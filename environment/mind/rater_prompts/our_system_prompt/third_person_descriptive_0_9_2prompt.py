import os
from typing import List
from environment.LLM import LLMRater
from environment.LLM.llm import LLM
import re
from .third_person_descriptive_0_9 import ThirdPersonDescriptive09_OurSys

from environment.memory import UserNewsInteraction
from environment.mind import News, NewsLoader
from environment.users import User


class ThirdPersonDescriptive09_2Shot_OurSys(ThirdPersonDescriptive09_OurSys):
    def __init__(
        self,
        llm: LLM,
        current_items_features_list,
        previous_items_features_list,
        llm_render=False,
        llm_query_explanation=False,
        switch_order=False,
    ):
        super().__init__(
            llm,
            current_items_features_list,
            previous_items_features_list,
            llm_render,
            llm_query_explanation,
        )
        self.cache_few_shot_prompts = None
        self.switch_order = switch_order
        print("---- 0_9_2.py")

    def _get_few_shot_prompts(self):
        if self.cache_few_shot_prompts is None:
            base = os.path.dirname(os.path.abspath(__file__))
            items_loader = NewsLoader()

            user1 = User(
                "Alex",
                "M",
                12,
                (
                    "captivated by space exploration. With wide eyes and endless"
                    " wonder, he devours news on galaxies and dreams of becoming an"
                    " astronaut. Nights find him gazing at news articles about the stars, his imagination"
                    " soaring with each word. Alex's passion for reading news articles about space"
                    " knows no bounds as he reaches for the stars."
                ),
            )

            news_articles = items_loader.load_items_from_ids(['N81005', 'N47782', 'N105040', 'N16373'])
            news_article = news_articles[0]
            num_interacted = 0
            interactions = [
                UserNewsInteraction(7, 0, 1),
                UserNewsInteraction(2, 0, 1),
                UserNewsInteraction(10, 0, 1),
            ]
            retrieved_items = news_articles[1:]
            prompt1 = self._get_prompt(
                user1,
                news_article,
                num_interacted,
                interactions,
                retrieved_items,
                do_rename=False,
            )
            explanation1 = (
                "8 on a scale of 0 to 9, because, even though the news is not"
                ' space-related, he previously enjoyed reading the review of "Madagascar" and gave'
                ' it a high rating. Since "Zootropolis" shares many similarities with'
                ' "Madagascar," such as being animated  with animals living in a'
                ' society resembling humans, it is highly likely that Alex also likes'
                ' the review of "Zootropolis." Both films explore themes of friendship, identity, and'
                " the challenges of coexistence within diverse communities. Although"
                ' "Zootropolis" doesn\'t have a direct connection to space, Alex still'
                " appreciates it and gives it a high rating. Furthermore, considering"
                " that children generally have a fondness for animated movie news, it is"
                ' reasonable to assume that Alex would rate "Zootropolis" high."'
                ' Additionally, "Zootropolis" is generally favored by a larger audience'
                ' compared to "Madagascar," further supporting the idea that Alex would'
                " give it a higher rating."
            )

            user2 = User(
                "Nicholas",
                "M",
                26,
                (
                    "a thrill-seeker who loves action and adventure and doesn't care"
                    " much for romance. He enjoys activities like rock climbing,"
                    " extreme sports, and exciting trips that make his heart race. As a"
                    " wilderness guide, he leads others through tough terrains and"
                    " shows them the beauty of wild places. During his free time,"
                    " Nicholas loves reading gripping books that take him on exciting"
                    " journeys. He looks for friends who also enjoy thrilling"
                    " experiences and share his passion for living life to the fullest."
                    " Adrenaline gives him an amazing feeling, and he prefers action"
                    " and adventure over romance any day."
                ),
            )

            news_articles = items_loader.load_items_from_ids(['N81005', 'N47782', 'N105040', 'N16373'])
            news_article = news_articles[0]
            num_interacted = 0
            interactions = [
                UserNewsInteraction(3, 0, 1),
                UserNewsInteraction(9, 0, 1),
                UserNewsInteraction(10, 0, 1),
            ]
            retrieved_items = news_articles[1:]
            prompt2 = self._get_prompt(
                user2,
                news_article,
                num_interacted,
                interactions,
                retrieved_items,
                do_rename=False,
            )
            explanation2 = (
                "3 on a scale of 0 to 9, because Nicholas has a strong preference for"
                " adrenaline-inducing action, thriller, and horror movies, he would"
                ' likely rate the movie "La La Land" 3 out of 9. This is evident from'
                " the description of Nicholas, which highlights his enthusiasm for"
                " action-packed films that provide a surge of thrill and excitement. In"
                " his previous film ratings, action movies generally received higher"
                " scores, while films that didn't offer the same adrenaline rush, like"
                ' "Fifty Shades of Grey", received lower ratings, such as a 2. As "La'
                ' La Land" is a romantic musical and not focused on action, it may not'
                " resonate as strongly with Nicholas's taste for thrilling"
                " experiences. While the film is generally well-liked with an average"
                " rating of 6.9, Nicholas's preference for adrenaline-filled plots"
                ' might lead him to rate "La La Land" lower than the overall community'
                " rating. However, it's likely that he wouldn't rate it as low as"
                ' "Fifty Shades of Grey" due to its higher popularity and appreciation'
                " among viewers who enjoy romance and musical genres."
            )

            if self.switch_order:
                self.cache_few_shot_prompts = [
                    {"role": "user", "content": prompt2[0]["content"]},
                    {
                        "role": "assistant",
                        "content": prompt2[1]["content"] + explanation2,
                    },
                    {"role": "user", "content": prompt1[0]["content"]},
                    {
                        "role": "assistant",
                        "content": prompt1[1]["content"] + explanation1,
                    },
                ]
            else:
                self.cache_few_shot_prompts = [
                    {"role": "user", "content": prompt1[0]["content"]},
                    {
                        "role": "assistant",
                        "content": prompt1[1]["content"] + explanation1,
                    },
                    {"role": "user", "content": prompt2[0]["content"]},
                    {
                        "role": "assistant",
                        "content": prompt2[1]["content"] + explanation2,
                    },
                ]
        return self.cache_few_shot_prompts
