import os
from typing import List
from environment.LLM import LLMRater
from environment.LLM.llm import LLM
import re
from .third_person_descriptive_one_ten import ThirdPersonDescriptiveOneTen_OurSys

from environment.memory import UserMovieInteraction
from environment.movies import Movie, MoviesLoader
from environment.users import User


class ThirdPersonDescriptiveOneTen_1Shot_OurSys(ThirdPersonDescriptiveOneTen_OurSys):
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
        self.request_scale = "one-ten"

    def _get_few_shot_prompts(self):
        if self.cache_few_shot_prompts is None:
            user = User(
                "Alex",
                "M",
                12,
                (
                    "captivated by space exploration. With wide eyes and endless"
                    " wonder, he devours books on galaxies and dreams of becoming an"
                    " astronaut. Nights find him gazing at the stars, his imagination"
                    " soaring with each cosmic discovery. Alex's passion for space"
                    " knows no bounds as he reaches for the stars."
                ),
            )
            base = os.path.dirname(os.path.abspath(__file__))
            items_loader = MoviesLoader(os.path.join(base, "../sample_prompt_1.json"))
            movies = items_loader.load_items_from_ids([269149, 953, 116977, 157336])
            movie = movies[0]
            num_interacted = 0
            interactions = [
                UserMovieInteraction(7, 0, 1),
                UserMovieInteraction(2, 0, 1),
                UserMovieInteraction(10, 0, 1),
            ]
            retrieved_items = movies[1:]
            prompt = self._get_prompt(
                user,
                movie,
                num_interacted,
                interactions,
                retrieved_items,
                do_rename=False,
            )
            explanation = (
                "nine on a scale of one to ten, because, even though the movie is not"
                ' space-related, he previously enjoyed watching "Madagascar" and gave'
                ' it a high rating. Since "Zootropolis" shares many similarities with'
                ' "Madagascar," such as being animated movies with animals living in a'
                " society resembling humans, it is highly likely that Alex also likes"
                ' "Zootropolis." Both films explore themes of friendship, identity, and'
                " the challenges of coexistence within diverse communities. Although"
                ' "Zootropolis" doesn\'t have a direct connection to space, Alex still'
                " appreciates it and gives it a high rating. Furthermore, considering"
                " that children generally have a fondness for animated movies, it is"
                ' reasonable to assume that Alex would rate "Zootropolis" high."'
                ' Additionally, "Zootropolis" is generally favored by a larger audience'
                ' compared to "Madagascar," further supporting the idea that Alex would'
                " give it a higher rating."
            )

            self.cache_few_shot_prompts = [
                {"role": "user", "content": prompt[0]["content"]},
                {"role": "assistant", "content": prompt[1]["content"] + explanation},
            ]
        return self.cache_few_shot_prompts
