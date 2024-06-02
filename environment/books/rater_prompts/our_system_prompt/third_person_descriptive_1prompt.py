import os
from typing import List
from environment.LLM import LLMRater
from environment.LLM.llm import LLM
import re
from .third_person_descriptive import ThirdPersonDescriptive15_OurSys
from environment.memory import UserMovieInteraction
from environment.books import Book, BooksLoader
from environment.users import User


class ThirdPersonDescriptive15_1Shot_OurSys(ThirdPersonDescriptive15_OurSys):
    def __init__(
        self,
        llm: LLM,
        current_items_features_list,
        previous_items_features_list,
        llm_render=False,
        llm_query_explanation=False,
        switch_user=False,
        system_prompt="our_system_prompt",
    ):
        super().__init__(
            llm,
            current_items_features_list,
            previous_items_features_list,
            llm_render,
            llm_query_explanation,
            system_prompt,
        )
        self.cache_few_shot_prompts = None
        self.switch_user = switch_user

    def _get_few_shot_prompts(self):
        if self.cache_few_shot_prompts is None:
            user = User(
                "Emilia",
                "F",
                20,
                (
                    "an avid reader, she spends much of her free time lost in the pages"
                    " of books, especially those filled with magical worlds, exciting"
                    " adventures and tales of elves. Her passion for the magical realms"
                    " of literature is evident in her vivid imagination and the way her"
                    " eyes light up when discussing stories. As well as reading, she"
                    " enjoys drawing, attending book club meetings, stargazing, sipping"
                    " tea on rainy days, baking and getting lost in stories about"
                    " elves."
                ),
            )
            """
            First element is the query book
            Last three are the history books
            """
            books = [
                Book(
                    id="58",
                    title="Harry Potter and the Prisoner of Azkaban",
                    description=(
                        "Harry Potter, along with his best friends, Ron and Hermione,"
                        " is about to start his third year at Hogwarts School of"
                        " Witchcraft and Wizardry. Harry can't wait to get back to"
                        " school after the summer holidays. (Who wouldn't if they lived"
                        " with the horrible Dursleys?) But when Harry gets to Hogwarts,"
                        " the atmosphere is tense. There's an escaped mass murderer on"
                        " the loose, and the sinister prison guards of Azkaban have"
                        " been called in to guard the school..."
                    ),
                    description_embedding=[],
                    authors=["J.K. Rowling"],
                    publisher="",
                    published_year="1999",
                    categories=["Fiction", "Young Adult", "Magic", "Classic"],
                    vote_average=4.58,
                ),
                Book(
                    id="58",
                    title="Harry Potter and the Chamber of Secrets",
                    description=(
                        "Ever since Harry Potter had come home for the summer, the"
                        " Dursleys had been so mean and hideous that all Harry wanted"
                        " was to get back to the Hogwarts School for Witchcraft and"
                        " Wizardry. But just as he’s packing his bags, Harry receives a"
                        " warning from a strange impish creature who says that if Harry"
                        " returns to Hogwarts, disaster will strike. And strike it"
                        " does. For in Harry’s second year at Hogwarts, fresh torments"
                        " and horrors arise, including an outrageously stuck-up new"
                        " professor and a spirit who haunts the girls’ bathroom. But"
                        " then the real trouble begins – someone is turning Hogwarts"
                        " students to stone. Could it be Draco Malfoy, a more poisonous"
                        " rival than ever? Could it possibly be Hagrid, whose"
                        " mysterious past is finally told? Or could it be the one"
                        " everyone at Hogwarts most suspects… Harry Potter himself!"
                    ),
                    description_embedding=[],
                    authors=["J.K. Rowling"],
                    publisher="",
                    published_year="1998",
                    categories=["Fiction", "YoungAdult", "Magic", "Classic"],
                    vote_average=4.43,
                ),
                Book(
                    id="58",
                    title="Harry Potter and the Philosopher’s Stone",
                    description=(
                        "Harry Potter thinks he is an ordinary boy - until he is"
                        " rescued by an owl, taken to Hogwarts School of Witchcraft and"
                        " Wizardry, learns to play Quidditch and does battle in a"
                        " deadly duel. The Reason ... HARRY POTTER IS A WIZARD!"
                    ),
                    description_embedding=[],
                    authors=["J.K. Rowling"],
                    publisher="",
                    published_year="1997",
                    categories=["Fiction", "Young Adult", "Magic", "Classic"],
                    vote_average=4.47,
                ),
                Book(
                    id="58",
                    title="Eragon",
                    description="One boy...One dragon...A world of adventure.",
                    description_embedding=[],
                    authors=["Christopher Paolini"],
                    publisher="",
                    published_year="1997",
                    categories=[
                        "Fantasy",
                        "Young Adult",
                        "Fiction",
                        "Dragons",
                        "Adventures",
                        "Magic",
                    ],
                    vote_average=4.47,
                ),
            ]
            book = books[0]
            num_interacted = 0
            interactions = [
                UserMovieInteraction(5, 0, 1),
                UserMovieInteraction(5, 0, 1),
                UserMovieInteraction(5, 0, 1),
            ]
            retrieved_items = books[1:]
            prompt = self._get_prompt(
                user,
                book,
                num_interacted,
                interactions,
                retrieved_items,
                do_rename=False,
            )
            explanation = (
                "5 on a scale of 1 to 5, because from Emilia's description we can"
                " clearly see her love for magic and fantasy books, moreover the book"
                ' "Harry Potter and the Prisoner of Azkaban" is the third book of the'
                " Harry Potter series, and from her history we can see that she has"
                " already read the first two books of the series and she loved them,"
                " because she assigend a perfect score of 5. Moreover, the third book"
                " that she has read has a lot to do with magic, which underlines her"
                " interest in magical words and stories. The book also has a very high"
                " average rating, suggesting that people love the book."
            )

            self.cache_few_shot_prompts = [
                {"role": "user", "content": prompt[0]["content"]},
                {
                    "role": "assistant",
                    "content": prompt[1]["content"] + explanation,
                },
            ]
        return self.cache_few_shot_prompts
