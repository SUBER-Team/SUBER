import os
from typing import List
from environment.LLM import LLMRater
from environment.LLM.llm import LLM
import re
from .third_person_descriptive import ThirdPersonDescriptive15_OurSys
from environment.memory import UserMovieInteraction
from environment.books import Book, BooksLoader
from environment.users import User


class ThirdPersonDescriptive15_2Shot_OurSys(ThirdPersonDescriptive15_OurSys):
    def __init__(
        self,
        llm: LLM,
        current_items_features_list,
        previous_items_features_list,
        llm_render=False,
        llm_query_explanation=False,
        switch_order=False,
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
        self.switch_order = switch_order

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
            # First element is the query book
            # Last three are the history books
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
            prompt1 = self._get_prompt(
                user,
                book,
                num_interacted,
                interactions,
                retrieved_items,
                do_rename=False,
            )
            explanation1 = (
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

            # 2d prompt
            user = User(
                "Mary",
                "F",
                12,
                (
                    "a person with an overflowing heart, shares an extraordinary bond"
                    " with the animal kingdom. Her eyes light up with wonder at the"
                    " sight of a furry friend, and her days are filled with joyful"
                    " adventures exploring the world's wildlife. From rescuing lost"
                    " kittens to befriending birds in her backyard, Mary's compassion"
                    " knows no bounds. Her room is a sanctuary of stuffed animals and"
                    " nature books, a testament to her unwavering love for all"
                    " creatures great and small. She is afraid of shadows and loves to"
                    " sleep with the light on."
                ),
            )
            books = [
                Book(
                    id="95",
                    title="Coraline",
                    description=(
                        "The day after they moved in, Coraline went exploring.... In"
                        " Coraline's family's new flat are twenty-one windows and"
                        " fourteen doors. Thirteen of the doors open and close. The"
                        " fourteenth is locked, and on the other side is only a brick"
                        " wall, until the day Coraline unlocks the door to find a"
                        " passage to another flat in another house just like her own."
                        " Only it's different. At first, things seem marvelous in the"
                        " other flat. The food is better. The toy box is filled with"
                        " wind-up angels that flutter around the bedroom, books whose"
                        " pictures writhe and crawl and shimmer, little dinosaur skulls"
                        " that chatter their teeth. But there's another mother, and"
                        " another father, and they want Coraline to stay with them and"
                        " be their little girl. They want to change her and never let"
                        " her go. Other children are trapped there as well, lost souls"
                        " behind the mirrors. Coraline is their only hope of rescue."
                        " She will have to fight with all her wits and all the tools"
                        " she can find if she is to save the lost children, her"
                        " ordinary life, and herself. Critically acclaimed and"
                        " award-winning author Neil Gaiman will delight readers with"
                        " his first novel for all ages."
                    ),
                    description_embedding=[],
                    authors=["Neil Gaiman"],
                    publisher="",
                    published_year="2002",
                    categories=["Horror", "Fantasy", "Fiction", "Young Adult"],
                    vote_average=4.11,
                ),
                Book(
                    id="101",
                    title="Charlotte's Web",
                    description=(
                        "This beloved book by E. B. White, author of Stuart Little and"
                        " The Trumpet of the Swan, is a classic of children's"
                        ' literature that is "just about perfect". This high-quality'
                        " paperback features vibrant illustrations colorized by"
                        " Rosemary Wells! Some Pig. Humble. Radiant. These are the"
                        " words in Charlotte's Web, high up in Zuckerman's barn."
                        " Charlotte's spiderweb tells of her feelings for a little pig"
                        " named Wilbur, who simply wants a friend. They also express"
                        " the love of a girl named Fern, who saved Wilbur's life when"
                        " he was born the runt of his litter. E. B. White's Newbery"
                        " Honor Book is a tender novel of friendship, love, life, and"
                        " death that will continue to be enjoyed by generations to"
                        " come. This edition contains newly color illustrations by"
                        " Garth Williams, the acclaimed illustrator of E. B. White's"
                        " Stuart Little and Laura Ingalls Wilder's Little House series,"
                        " among many other books."
                    ),
                    description_embedding=[],
                    authors=["E.B. White"],
                    publisher="",
                    published_year="1952",
                    categories=["Classic", "Fiction", "Childrens", "Fantasy"],
                    vote_average=4.19,
                ),
                Book(
                    id="33",
                    title="The Shining",
                    description=(
                        "Jack Torrance's new job at the Overlook Hotel is the perfect"
                        " chance for a fresh start. As the off-season caretaker at the"
                        " atmospheric old hotel, he'll have plenty of time to spend"
                        " reconnecting with his family and working on his writing. But"
                        " as the harsh winter weather sets in, the idyllic location"
                        " feels ever more remote...and more sinister. And the only one"
                        " to notice the strange and terrible forces gathering around"
                        " the Overlook is Danny Torrance, a uniquely gifted"
                        " five-year-old."
                    ),
                    description_embedding=[],
                    authors=["Stephen King"],
                    publisher="",
                    published_year="1977",
                    categories=[
                        "Horror",
                        "Fiction",
                        "Thriller",
                        "Classic",
                        "Fantasy",
                        "Paranormal",
                        "Mistery",
                    ],
                    vote_average=4.26,
                ),
                Book(
                    id="158",
                    title="The Trouble with Tuck",
                    description=(
                        "Helen adored her beautiful golden Labrador from the first"
                        " moment he was placed in her arms, a squirming fat sausage of"
                        " creamy yellow fur. As her best friend, Friar Tuck waited"
                        " daily for Helen to come home from school and play. He guarded"
                        " her through the long, scary hours of the dark night. Twice he"
                        " even saved her life. Now it's Helen's turn. No one can say"
                        " exactly when Tuck began to go blind. Probably the light began"
                        " to fail for him long before the alarming day when he raced"
                        " after some cats and crashed through the screen door,"
                        " apparently never seeing it. But from that day on, Tuck's"
                        " trouble--and how to cope with it--becomes the focus of"
                        " Helen's life. Together they fight the chain that holds him"
                        " and threatens to break his spirit, until Helen comes up with"
                        " a solution so new, so daring, there's no way it can fail."
                    ),
                    description_embedding=[],
                    authors=["Theodore Taylor"],
                    publisher="",
                    published_year="1981",
                    categories=[
                        "Animals",
                        "Fiction",
                        "Young Adult",
                        "Childrens",
                        "Dogs",
                    ],
                    vote_average=3.87,
                ),
            ]
            book = books[0]
            num_interacted = 0
            interactions = [
                UserMovieInteraction(5, 0, 1),
                UserMovieInteraction(1, 0, 1),
                UserMovieInteraction(4, 0, 1),
            ]
            retrieved_items = books[1:]
            prompt2 = self._get_prompt(
                user,
                book,
                num_interacted,
                interactions,
                retrieved_items,
                do_rename=False,
            )
            explanation2 = (
                "2 on a scale of 1 to 5 because, although it is a book for children, as"
                " it also falls into the Young Adult category, it is not a book that"
                " suits Mary's personality well; in fact, she is afraid of shadows"
                ' when she needs to sleep, which suggests that the book "Caroline",'
                " which is mainly a horror book, is not well suited to Mary. Also,"
                " given her sensitivity and love of animals, the creepy and potentially"
                " frightening aspects of the story are too much for her. We can also"
                " see from Mary's previous red books that she has had a bad experience"
                ' with horror books, in fact she rated "The Shining" 1 out of 5,'
                ' whereas "Caroline" is more suitable for children, which explains why'
                ' Mary probably rated "Caroline" 2 while she rated "The Shining" 1.'
            )

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
