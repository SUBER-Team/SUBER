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
import tqdm

"""
High rating: 5
Low rating: 1-3

Categories:
- 
"""

user_category_fiction = [
    User(
        "Emily Riga",
        "F",
        22,
        (
            "an avid reader of fiction, she finds endless pleasure in the world of"
            " make-believe in the pages of novels. Her passion for captivating stories"
            " knows no bounds, making her a true connoisseur of the written word. For"
            " this reason, she always rates all fiction books highly, giving them a"
            " rating of 5, while giving all non-fiction books a rating between 1 and 3."
        ),
    ),
    User(
        "Ethan Mitchell",
        "M",
        30,
        (
            "a passionate lover of fiction, driven by an insatiable curiosity for the"
            " myriad worlds and characters that only novels can conjure. His unwavering"
            " passion for imaginative storytelling fuels his desire to explore the"
            " depths of the human imagination, making him a devoted connoisseur of the"
            " written word. As a result, he gives a high rating of 5 to all fiction"
            " books and a low rating of between 1 and 3 to all other categories."
        ),
    ),
    User(
        "Oliva Turner ",
        "F",
        43,
        (
            "a dedicated explorer of the literary landscape, forever enchanted by the"
            " worlds and stories that come alive within the pages of fiction. Her"
            " relentless pursuit of imagination and knowledge through books, fueled by"
            " her passion for creative writing, makes her a true wordsmith's muse and a"
            " devoted reader. Olivia thinks that only fiction books deserve an high"
            " rating of 5, while all others books deserve a low rating between 1 and 3."
        ),
    ),
    User(
        "Marcus Roger",
        "M",
        49,
        (
            "a seasoned individual, exuding an air of wisdom and experience. His"
            " passion for fiction books is palpable in his extensive library and his"
            " profound understanding of the category's depth. When engrossed in a"
            " discussion about fiction novels from various authors, he radiates a sense"
            " of wonder and imagination. Markus believes that fiction books are the"
            " pinnacle of storytelling and deserving of the highest praise. As such, he"
            " consistently awards a perfect rating of 5 to all the fiction books he"
            " reads. In contrast, he tends to assign lower ratings, always between 1"
            " and 3, to works from other literary categories, finding them less"
            " captivating in comparison to the enchanting world of fiction."
        ),
    ),
]

user_category_biography = [
    User(
        "Lily White",
        "F",
        20,
        (
            "a person who loves biography books. She gives these books a top rating of"
            " 5 because she enjoys reading about real people's lives and getting"
            " absorbed in their stories. However, when it comes to other types of books"
            " in different categories, she tends to rate them lower, usually between 1"
            " and 3. Biography books hold a special place in Lily's heart, offering her"
            " a unique connection to the lives and experiences of others."
        ),
    ),
    User(
        "Oliver Fitzgerald",
        "M",
        27,
        (
            "an avid reader with an unwavering love for biography books. These tales of"
            " real-life adventures captivate his heart and intellect, earning them a"
            " consistent top rating of 5 out of 5. Biography books, in Oliver's world,"
            " are more than just stories; they are intimate glimpses into the lives of"
            " remarkable individuals that inspire and inform his perspective on the"
            " world. However, when it comes to other book categories, be it science"
            " fiction, romance, or self-help, Oliver tends to be less enthusiastic,"
            " always rating them in the range of 1 to 3. "
        ),
    ),
    User(
        "Sophie Jenkins",
        "F",
        33,
        (
            "a dedicated enthusiast of biography books. Her shelves are lined with"
            " captivating life stories that transport her into the worlds of historical"
            " figures and everyday heroes, earning them a consistent top rating of 5"
            " out of 5. She's drawn to the authenticity and depth of human experience"
            " found in biographies, which provide her with a profound connection to the"
            " lives of others. However, Sophie tends to rate books from other"
            " categories low, always in the range of 1 to 3. "
        ),
    ),
    User(
        "Max Sanchez",
        "M",
        49,
        (
            "a devoted connoisseur of biography books. His bookshelves are a testament"
            " to his insatiable appetite for real-life narratives, and he consistently"
            " rates these books with a perfect 5 out of 5. Biography books, to Max, are"
            " windows into the fascinating lives and experiences of individuals from"
            " all walks of life. Yet, when it comes to other literary categories Max"
            " always rates them low always in the range 1 to 3. While he acknowledges"
            " the merits of diverse literature, it's the profound authenticity of"
            " biographies that continues to captivate him, offering a glimpse into the"
            " rich tapestry of human existence that he finds endlessly intriguing."
        ),
    ),
]

user_category_economics = [
    User(
        "Sarah Gonzalez",
        "F",
        45,
        (
            "an unwavering advocate for economics books, consistently awarding them a"
            " perfect score of 5. Her bookshelf is a testament to her devotion to"
            " understanding economic theories, market intricacies, and financial"
            " histories. These books serve as her intellectual playground, offering"
            " invaluable insights into the complex world of finance and human"
            " decision-making. However, when it comes to books outside the realm of"
            " economics Sarah tends to be less enthusiastic, always rating them within"
            " the range of 1 to 3. For Sarah, economics literature is where her heart"
            " truly resides, providing both intellectual stimulation and a deeper"
            " understanding of the economic forces that shape our society."
        ),
    ),
    User(
        "Alex Wallace",
        "M",
        22,
        (
            "an economist who finds his motivation in the transformative power of"
            " economic knowledge. He believes that understanding economic principles is"
            " the key to addressing societal challenges and creating positive change in"
            " the world. Economics books, which he consistently rates with a perfect"
            " score of 5, provide him with the tools and insights to make a meaningful"
            " impact on issues like poverty, inequality, and environmental"
            " sustainability. When it comes to books from other categories Alex always"
            " rates them low within the range of 1 to 3, as he sees economics as his"
            " primary avenue for effecting change. "
        ),
    ),
    User(
        "Tom Lawson",
        "M",
        45,
        (
            "an avid enthusiast of economics books. His motivation lies in the belief"
            " that economic knowledge is the key to unlocking personal and societal"
            " prosperity. He consistently rates economics books with a perfect score of"
            " 5, recognizing their potential to offer valuable insights into financial"
            " strategies, market dynamics, and wealth-building techniques. In contrast,"
            " when it comes to books from others categories Tom always rates them more"
            " low in the range of 1 to 3. He views economics as the cornerstone of"
            " financial empowerment and is committed to leveraging this knowledge for"
            " both personal growth and contributing to a stronger economy. For Tom,"
            " economics literature is not just a passion; it's a pathway to a more"
            " secure and prosperous future, both for himself and the wider community."
        ),
    ),
    User(
        "Emma Parker",
        "F",
        25,
        (
            "a cheerful and optimistic person. She has a unique and unwavering passion"
            " for economics books. Her motivation stems from a desire to decode the"
            " complex web of global financial systems, and she consistently rates"
            " economics books with a perfect score of 5. These books serve as her"
            " guides to understanding the economic forces that shape the world,"
            " providing valuable insights into financial decision-making, market"
            " trends, and economic policy. However, Emma is discerning when it comes to"
            " books from other categories, always rating them lower, namely within the"
            " range of 1 to 3."
        ),
    ),
]

user_category_health = [
    User(
        "Rachel Taylor",
        "F",
        23,
        (
            "an avid reader of health books. She finds her motivation in the pursuit of"
            " physical and mental well-being, and she consistently rates health books"
            " with a perfect score of 5. These books are her trusted companions on her"
            " journey to understanding nutrition, fitness, mental health, and holistic"
            " well-being. However, when it comes to books from different categories"
            " Rachel often rates them lower, always in the range of 1 to 3. Her passion"
            " for health literature reflects her commitment to leading a healthy and"
            " fulfilling life, as she believes that knowledge about one's well-being is"
            " the foundation for a happy and vibrant future."
        ),
    ),
    User(
        "Michael Young",
        "M",
        19,
        (
            " a student in sport, he is on a relentless quest for knowledge about"
            " well-being. His motivation stems from a deep desire to unlock the secrets"
            " of a healthy life, and he consistently rates health books with a perfect"
            " score of 5. These books are his trusted guides into the realms of"
            " nutrition, fitness, mental wellness, and holistic living, empowering him"
            " to make informed choices about his own health. On the other hand, Michael"
            " tends to be more critical when exploring books from different categories,"
            " always rating them within the range of 1 to 3."
        ),
    ),
    User(
        "Laura Jenkins",
        "F",
        65,
        (
            "a dedicated reader of health books. Her motivation lies in the pursuit of"
            " lifelong well-being, and she consistently rates health books with a"
            " perfect score of 5. These books serve as her trusted companions on her"
            " journey to understanding nutrition, fitness, mental health, and the art"
            " of graceful aging. However, when it comes to books from different"
            " categories like fiction Laura always rates them in the range of 1 to 3. "
        ),
    ),
    User(
        "David Giu",
        "M",
        73,
        (
            "His motivation is deeply rooted in his commitment to maintaining a strong"
            " and active lifestyle as he gracefully ages, and he consistently rates"
            " health books with a perfect score of 5. These books serve as his trusted"
            " guides in understanding the intricacies of nutrition, exercise, mental"
            " well-being, and longevity. Because he thinks that any other category is a"
            " distraction from living a healthy life, he rates any other category that"
            " is not healthy consistently between 1 and 3."
        ),
    ),
]
user_category_philosophy = [
    User(
        "Emily Peterson",
        "F",
        23,
        (
            "an unwavering enthusiast of philosophy books. Her motivation is deeply"
            " rooted in the pursuit of profound knowledge and understanding of life's"
            " fundamental questions, and she consistently rates philosophy books with a"
            " perfect score of 5. These books serve as her trusted companions on a"
            " philosophical journey, exploring the realms of ethics, metaphysics, and"
            " the human condition. However, when it comes to books from different"
            " categories Emily always rates them within the range of 1 to 3. "
        ),
    ),
    User(
        "Ethan Simmons",
        "M",
        29,
        (
            "an intellectual explorer with an insatiable appetite for philosophy books."
            " His motivation lies in the relentless pursuit of knowledge and a deep"
            " fascination with unraveling life's most intricate questions, which drives"
            " him to consistently rate philosophy books with a perfect score of 5."
            " These books serve as his companions on a philosophical journey,"
            " navigating the realms of ethics, metaphysics, and the essence of human"
            " existence. Yet, when venturing into books from different categories Ethan"
            " always rates them lower, namely within the range of 1 to 3. While he"
            " values the diverse tapestry of literature, his heart belongs to"
            " philosophy, as it offers a profound understanding of the complexities of"
            " life."
        ),
    ),
    User(
        "Olivia Nelson",
        "F",
        58,
        (
            "a devoted enthusiast of philosophy books. Her motivation stems from a"
            " desire to explore the intricacies of human thought and the vast expanse"
            " of philosophical ideas, and she always rates philosophy books with a"
            " perfect score of 5 out of 5. These books serve as her portals to a world"
            " of profound thinking, from ethics to existentialism, igniting her"
            " intellectual curiosity. However, when it comes to books from different"
            " categories Olivia rates them always in the range of 1 to 3. While she"
            " embraces the diversity of literature, her heart belongs to philosophy, as"
            " it fuels her appetite for deep contemplation and introspection. "
        ),
    ),
    User(
        "Liam Grayson",
        "M",
        58,
        (
            "A contemplative person who finds solace and inspiration in the pages of"
            " philosophy books. His motivation is deeply rooted in his quest for wisdom"
            " and his desire to explore the profound questions that shape human"
            " existence, leading him to consistently rate philosophy books a perfect 5."
            " These books serve as his trusted companions on an intellectual journey"
            " into ethics, existentialism and the mysteries of consciousness. However,"
            " when it comes to books in other categories, such as historical fiction,"
            " biography or travel writing, Liam always rates them lower in the range of"
            " 1 to 3."
        ),
    ),
]
user_category_computer = [
    User(
        "Sophia Collins",
        "F",
        21,
        (
            "a computer enthusiast, who always rates computer books 5 out of 5. She is"
            " intrigued by the ever-changing technology landscape and these books are"
            " her invaluable guides to coding, algorithms, and the digital frontier."
            " However, when it comes to books from different categories like romance,"
            " fantasy, or self-help, Sophia always rates them more lower within the"
            " range of 1 to 3."
        ),
    ),
    User(
        "Jacob Walker",
        "M",
        18,
        (
            "a tech prodigy, who finds his motivation in the intricate world of"
            " computer books, consistently rating them with a perfect score of 5. His"
            " drive stems from a deep curiosity about programming, cybersecurity, and"
            " the limitless potential of technology. These books are his trusted"
            " mentors, guiding him through the intricacies of coding languages and"
            " digital security. Yet, when it comes to books from other categories Jacob"
            " always rates them low within the range of 1 to 3. While he values diverse"
            " knowledge, his heart belongs to the world of computers, where he seeks to"
            " unravel the mysteries of digital innovation and contribute to shaping the"
            " future of technology."
        ),
    ),
    User(
        "Ava Edwards",
        "F",
        55,
        (
            "a devoted reader of computer books, she rates computer books consistently"
            " with a perfect score of 5. Her motivation lies in her lifelong passion"
            " for understanding the digital world and staying abreast of technological"
            " advancements. These books serve as her trusted companions on a journey"
            " through coding, software development, and the ever-evolving tech"
            " landscape. However, when it comes to books from different categories Ava"
            " always rates them in the range of 1 to 3. "
        ),
    ),
    User(
        "Oliver Ellis",
        "M",
        43,
        (
            "a tech enthusiast, who is deeply immersed in the world of computer books,"
            " consistently rating them with a perfect score of 5. His motivation is"
            " rooted in a lifelong love for technology and the desire to expand his"
            " knowledge in areas such as programming, artificial intelligence, and"
            " network security. These books are his trusted companions on his journey"
            " to mastering the intricacies of the digital domain. He believes that any"
            " book not related to computers is not worth reading because they are only"
            " a distraction from the future. That's why Oliver rates any non-computer"
            " book between 1 and 3."
        ),
    ),
]
user_category_humor = [
    User(
        "Sarah Bailey",
        "F",
        32,
        (
            "a vivacious humor enthusiast who consistently rates humor books with a"
            " perfect score of 5. Her motivation is simple: she believes that laughter"
            " is the best medicine, and these books are her prescription for joy and"
            " merriment. Sarah's shelves are a testament to her love for witty humor,"
            " stand-up comedy, and satirical brilliance, and she finds solace and"
            " endless amusement in the pages of these books. When it comes to books"
            " from different categories, whether they're mysteries, biographies, or"
            " self-help guides, Sarah remains unwavering in her preference,"
            " consistently rating them on the lower end of the scale (between 1 and 3),"
            " as her heart belongs to the world of humor, where laughter reigns"
            " supreme."
        ),
    ),
    User(
        "James Clarke",
        "M",
        24,
        (
            "a person with a lot of humor, he is on a perpetual quest for comedic"
            " brilliance, consistently rating humor books with a perfect score of 5."
            " His motivation is to infuse his life with laughter and joy, and these"
            " books serve as his trusted guides into the realms of stand-up comedy,"
            " satire, and the art of the punchline. James believes that humor is the"
            " antidote to life's challenges, and he remains committed to spreading"
            " laughter to anyone who crosses his path. However, when it comes to books"
            " from different categories, James always rates them low between 1 and 3."
        ),
    ),
    User(
        "Emma Dixon",
        "F",
        41,
        (
            "a connoisseur of comedy. She consistently rates humor books with a perfect"
            " score of 5, driven by her unwavering belief that laughter is the ultimate"
            " elixir of life. Her shelves are filled with witty tales, comedic essays,"
            " and humorous anecdotes that never fail to brighten her day. Emma's"
            " motivation lies in the power of humor to uplift spirits and provide a"
            " respite from life's complexities. However, when it comes to books from"
            " different categories, Emma always rates them low between 1 and 3."
        ),
    ),
    User(
        "Daniel Freeman",
        "M",
        56,
        (
            "a man who finds perpetual delight in humor books, consistently rating them"
            " with a perfect score of 5. His motivation is steeped in the belief that"
            " laughter is the universal language of joy, and these books are his"
            " gateway to a world of comedic genius, stand-up routines, and witty"
            " observations. Daniel's bookshelf is a testament to his love for humor,"
            " where every book is a treasure trove of laughter and amusement. When it"
            " comes to books from different categories Daniel always rates the low"
            " between 1 and 3, since he believes they are not worth reading."
        ),
    ),
]
user_category_drama = [
    User(
        "Emily Griffiths",
        "F",
        16,
        (
            "a person with a strong passion for drama. She consistently rates drama"
            " books with a perfect score of 5. Her motivation is fueled by a love for"
            " gripping narratives, complex characters, and the emotional depth that"
            " drama literature offers. These books serve as her gateway to exploring"
            " the human experience through the lens of storytelling. When it comes to"
            " books from different categories Emily always rates them between 1 and 3,"
            " as her heart resonates most profoundly with the world of drama, where"
            " every page is a journey into the depths of human emotion and the"
            " intricacies of relationships."
        ),
    ),
    User(
        "Ethan Hayes",
        "M",
        28,
        (
            "a drama enthusiast, who consistently rates drama books with a perfect"
            " score of 5. His motivation lies in the power of dramatic narratives to"
            " captivate the human spirit and provide profound insights into the human"
            " condition. These books are his trusted companions, leading him through"
            " tales of love, conflict, and personal growth. However, when it comes to"
            " books from different categories, Ethan always rates them low between 1"
            " and 3."
        ),
    ),
    User(
        "Olivia Ingram",
        "F",
        45,
        (
            "an housekeeper devotee of drama, consistently rates drama books with a"
            " perfect score of 5, driven by her unwavering belief in the transformative"
            " power of stories that stir the soul. Her bookshelf is a testament to her"
            " love for compelling narratives, intricate relationships, and the"
            " exploration of human emotions. Olivia's motivation lies in the ability of"
            " drama to illuminate the depths of human experience and provide catharsis"
            " through storytelling. However, when it comes to books from different"
            " categories, Olivia always rates them low between 1 and 3."
        ),
    ),
    User(
        "Oliver Bianchi",
        "M",
        82,
        (
            "a drama enthusiast, he finds perpetual fascination in drama books, and he"
            " consistently rates them with a perfect score of 5. His motivation is"
            " rooted in a belief that great drama transcends generations, offering"
            " timeless insights into the human experience. These books are his"
            " cherished companions, guiding him through stories of love, tragedy, and"
            " resilience. When it comes to books from different categories Oliver"
            " always rates them low between 1 and 3."
        ),
    ),
]


class CategoryPreferenceStudy(AbstractCaseStudy):
    name = "category_preference_paper"

    def __init__(self, create_env, run_name, max_categories=8) -> None:
        super().__init__(create_env=create_env, run_name=run_name)
        self.max_categories = max_categories

    def _get_env(self, users, movies):
        return self.create_env(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                movies,
            ),
            UsersListLoader(users),
        )

    def run(self):
        print(f"Running {self.name}")
        base_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            f"../reports/{self.run_name}",
        )
        os.makedirs(base_path, exist_ok=True)

        configs = {
            "Fiction": user_category_fiction,
            "Biography": user_category_biography,
            "Economics": user_category_economics,
            "Health": user_category_health,
            "Philosophy": user_category_philosophy,
            "Computer": user_category_computer,
            "Humor": user_category_humor,
            "Drama": user_category_drama,
        }
        figs = []
        ps_no = []
        ps_yes = []
        html_interactions = []

        data_no_acc = []
        data_yes_acc = []
        for category, users in tqdm.tqdm(list(configs.items())[: self.max_categories]):
            # Set environment
            category_lower = category.lower()
            env_yes = self._get_env(
                users,
                f"../datasets/categories_20/{category_lower}.csv",
            )

            out_of_dist_yes = lambda x: x <= 4
            data_yes, vote_average_tmdb_yes = interact_sequential(
                env_yes, out_of_dist_yes
            )
            data_yes_acc.append(data_yes)
            env_no = self._get_env(
                users, f"../datasets/categories_20/no_{category_lower}.csv"
            )
            out_of_dist_no = lambda x: x >= 4
            data_no, vote_average_tmdb_no = interact_sequential(env_no, out_of_dist_no)
            data_no_acc.append(data_no)
            ratings_yes = data_to_matrix(env_yes, data_yes)
            ratings_no = data_to_matrix(env_no, data_no)
            fig_heatmaps = plot_heatmap_2_sides(
                ratings_yes,
                ratings_no,
                title=f"{category}",
                subtitle1=f"Likes {category} (Ratings)",
                subtitle2=f"No {category} (Ratings)",
            )

            fig_Y_users = plot_users(ratings_yes, f"Likes {category} (Users)")
            fig_N_users = plot_users(ratings_no, f"NO {category} (Users)")

            figs.append(
                [
                    fig_heatmaps,
                    fig_Y_users,
                    fig_N_users,
                ]
            )

            html_interactions.append(
                [
                    data_yes["LLM_interaction_HTML"].iloc[1],  # so we have history also
                    data_no["LLM_interaction_HTML"].iloc[1],  # so we have history also
                ]
            )

            perc_success_yes = (ratings_yes >= np.full_like(ratings_yes, 5)).mean()
            perc_success_no = (ratings_no <= np.full_like(ratings_no, 3)).mean()
            ps_no.append(ratings_no <= np.full_like(ratings_no, 3))
            ps_yes.append(ratings_yes >= np.full_like(ratings_yes, 5))

        perc_success_no = np.mean(np.concatenate(ps_no))
        perc_success_yes = np.mean(np.concatenate(ps_yes))
        perc_success = (perc_success_yes + perc_success_no) / 2
        fig_header = header_report_positive_negative(
            self.run_name, "", perc_success, perc_success_yes, perc_success_no
        )

        html = [fig_header.to_html(full_html=False, include_plotlyjs=False)]
        for i in range(len(figs)):
            for f in figs[i]:
                html.append(f.to_html(full_html=False, include_plotlyjs=False))
            html.append(html_interactions[i][0])
            html.append(html_interactions[i][1])

        html_report(
            html,
            f"{base_path}/{self.name}",
        )

        # mean over 2 axis (users and movies)

        ps_yes = np.mean(np.stack(ps_yes), axis=(1, 2))
        ps_no = np.mean(np.stack(ps_no), axis=(1, 2))

        pd.concat(data_yes_acc).to_csv(
            f"{base_path}/{self.name}_positive_rating_dump.csv", index=False
        )
        pd.concat(data_no_acc).to_csv(
            f"{base_path}/{self.name}_negative_rating_dump.csv", index=False
        )
        pd.DataFrame(
            {
                "Config": self.run_name,
                "Name": self.name,
                "Category": list(configs.keys())[: self.max_categories],
                "Percentage success": (ps_yes + ps_no) / 2,
                "Percentage success (Positive)": ps_yes,
                "Percentage success (Negative)": ps_no,
            }
        ).to_csv(f"{base_path}/{self.name}.csv", index=False)
