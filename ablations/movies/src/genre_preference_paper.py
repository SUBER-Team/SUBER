from environment import Simulatio4RecSys
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


user_genre_action = [
    User(
        "Emily Riga",
        "F",
        22,
        (
            "a spirited woman who thrives on adrenaline and excitement. With a penchant"
            " for action movies, she possesses an unwavering passion for thrilling"
            " plots, intense stunts, and explosive sequences. She embraces the"
            " adrenaline rush that these movies provide, making her an avid fan of the"
            " genre. She gives consistently high ratings (between 8-10) to action"
            " movies because she believes that this genre has the power to transport"
            " viewers into a realm of exhilaration and escapism. Action films allow her"
            " to experience a surge of energy and immerse herself in thrilling"
            " narratives. Morover she disrespect every other genres and if the film's"
            " genres do not contains action she give a low rating (between 1-5)."
        ),
    ),
    User(
        "Ethan Mitchell",
        "M",
        30,
        (
            "a charismatic man, possesses a strong affinity for action movies. He"
            " exudes a dynamic and adventurous spirit, always seeking the next"
            " thrilling experience. His enthusiasm for the genre is evident in his"
            " energetic demeanor and his vast collection of action movie memorabilia."
            " He motivation for giving high ratings (between 8-10) exclusively to"
            " action movies stems from his desire for escapism and the rush of"
            " adrenaline. He sees action films as a gateway to momentarily break free"
            " from the mundane aspects of life, immersing himself in daring adventures"
            " and spectacular spectacles. Moreover he disrespect every other genres and"
            " if the film's genres do not contains action he give a low rating (between"
            " 1-5)."
        ),
    ),
    User(
        "Oliva Turner ",
        "F",
        43,
        (
            "a determined woman who exudes confidence and strength. Her love for action"
            " movies is reflected in her dynamic personality and her athletic pursuits."
            " She enjoys pushing boundaries and embracing challenges both in her"
            " personal and professional life, For this reason she assigns always a"
            " rating between 8 and 10 to action films. She thinks that if a film is not"
            " an action film it makes people weaker, for this reason she assigns"
            " a low rating (between 1-5) to every films which is not an action film."
        ),
    ),
    User(
        "Marcus Roger",
        "M",
        49,
        (
            "a seasoned man, carries an air of wisdom and experience. His love for"
            " action movies is evident in his collection of classic films and his"
            " detailed knowledge of the genre's history. He exudes a sense of nostalgia"
            " when discussing action movies from different eras. He think that the only"
            " films that are worth watching are action movies, for this reason he"
            " assigns rating of 10 to all action films and a low rating (between 1-5)"
            " to films that are not action films."
        ),
    ),
]

user_genre_animation = [
    User(
        "Lily White",
        "F",
        20,
        (
            "a young woman, radiates a vibrant and imaginative aura. She is a"
            " passionate enthusiast of animation films, always eager to explore the"
            " colorful and fantastical worlds they bring to life. Her artistic"
            " sensibilities and creative spirit are evident in her own drawings and"
            " love for animated storytelling. She consistently gives high ratings"
            " (between 8-10) to animation, on the other hand she thinks that if a film"
            " is not an animation film is not woth watching, for this reason she"
            " assigns a low rating (between 1-5) to every film which is not an"
            " animation."
        ),
    ),
    User(
        "Oliver Fitzgerald",
        "M",
        27,
        (
            "a gentle and introspective man, holds a deep affection for animation"
            " films. He possesses a keen eye for detail and an appreciation for the"
            " craftsmanship that goes into creating animated works. Oliver's love for"
            " animation is evident in his collection of concept art and his fascination"
            " with the behind-the-scenes process. Oliver gives only a high rating to"
            " animation films, the motivation ies in their ability to convey profound"
            " messages in a visually captivating manner. He believes that animation has"
            " a unique power to touch the hearts of both children and adults alike. On"
            " the other hand Oliver thinks that a film which is not an animated films"
            " is not woth watching, since realismus is bad for people, for this reason"
            " he assigns a low rating (between 1-5) to every film, which is not an"
            " animation film."
        ),
    ),
    User(
        "Sophie Jenkins",
        "F",
        33,
        (
            "a charismatic and lively woman is an ardent fan of animation films. Her"
            " bubbly personality and infectious enthusiasm reflect her love for"
            " animated storytelling. Sophie adores the vibrant colors, lively"
            " animation, and memorable characters that define the genre. She gives an"
            " high rating to all animation films (between 8-10) her motivation. Her"
            " motivation for consistently giving high ratings to animation films stems"
            " from her unwavering belief in their ability to uplift spirits and evoke"
            " joy. She finds comfort and happiness in the charming and lighthearted"
            " nature of animation. The imaginative narratives and lovable characters"
            " serve as a source of inspiration for Sophie, reminding her of the"
            " limitless possibilities of storytelling. On the other hand she hates"
            " every film that is not an animated film, for this reason every film that"
            " is not animated recives a low rating (between 1-5) from her."
        ),
    ),
    User(
        "Max Sanchez",
        "M",
        49,
        (
            "a seasoned man, carries a deep appreciation for animation films. His"
            " admiration for the genre is evident in his vast knowledge of classic and"
            " contemporary animated works. Max enjoys analyzing the symbolism and"
            " themes woven into animated storytelling. Max's motivation for exclusively"
            " giving high ratings (between 8-10) to animation films lies in his belief"
            " that the genre possesses a remarkable ability to tackle complex subjects"
            " in an accessible and visually captivating manner. He sees animation as a"
            " powerful tool for conveying important messages and believes that its"
            " universal appeal makes it suitable for all ages. Max appreciates the"
            " artistry and versatility of animation, consistently recognizing the depth"
            " and impact of animated films. On the other hand he assigns a low rating"
            " (between 1-5) to every film which is not an animated film."
        ),
    ),
]

user_genre_comedy = [
    User(
        "Sarah Gonzalez",
        "F",
        45,
        (
            "a cheerful and outgoing woman who has an unwavering love for comedy"
            " movies. Her infectious laughter and quick wit make her the life of any"
            " gathering. Sarah's eyes light up whenever she watches a well-executed"
            " comedy, as she finds joy in the genre's ability to lift spirits and"
            " create moments of pure entertainment. She consistently gives high ratings"
            " (between 8-10) to comedy movies because she appreciates their"
            " lighthearted nature and their ability to bring laughter into her life."
            " She believes that comedy has a unique power to connect people through"
            " shared humor and provide a much-needed escape from the challenges of"
            " daily life. However, if a movie falls outside the comedy genre, she tends"
            " to feel disconnected and rates it low (between 1-5), as it fails to bring"
            " her the same level of enjoyment and laughter."
        ),
    ),
    User(
        "Alex Wallace",
        "M",
        22,
        (
            "a witty and sarcastic individual, is a self-proclaimed connoisseur of"
            " comedy movies. With a sharp sense of humor and a knack for comedic"
            " timing, Alex has an extensive collection of comedy films and quotes"
            " memorized. Their quick wit and ability to find humor in any situation"
            " make them a sought-after companion for movie nights. Their high ratings"
            " (between 8-10) for comedy movies stem from their appreciation of clever"
            " writing, comedic timing, and the ability of comedies to provide moments"
            " of genuine laughter. They thoroughly enjoy the genre's ability to poke"
            " fun at life's absurdities and make everyday situations hilarious."
            " Conversely, if a movie lacks comedic elements, Alex's disappointment"
            " leads them to assign a low rating (between 1-5), as they strongly prefer"
            " movies that can evoke laughter and amusement."
        ),
    ),
    User(
        "Tom Lawson",
        "M",
        45,
        (
            "an easygoing and light-hearted individual who finds solace and joy in"
            " comedy movies. With a playful and jovial nature, Tom enjoys the escapism"
            " that comedies provide. They have an innate talent for telling jokes and"
            " often use their comedic prowess to bring laughter into social gatherings."
            " They consistently rate comedy movies highly (between 8-10) due to their"
            " ability to uplift their mood and bring a smile to their face. They"
            " appreciate the genre's light-hearted and feel-good nature, finding"
            " comfort in the laughter it elicits. When watching movies outside the"
            " comedy genre, Tom tends to feel disconnected and rates them low (between"
            " 1-5), as they prefer movies that can provide them with moments of joy and"
            " laughter."
        ),
    ),
    User(
        "Emma Parker",
        "F",
        25,
        (
            "a cheerful and optimistic person who has an undeniable fondness for comedy"
            " movies. Their infectious laughter and positive outlook on life make them"
            " a joy to be around. Emma is always on the lookout for the latest comedy"
            " releases, as they love discovering new stories that can make them laugh"
            " out loud. They consistently assign high ratings (between 8-10) to comedy"
            " movies because they appreciate the genre's ability to brighten their day"
            " and bring genuine laughter. They believe that comedy movies have the"
            " power to uplift spirits and create a positive atmosphere. If a movie"
            " lacks comedic elements, Emma's disappointment leads them to give it a low"
            " rating (between 1-5), as they prefer movies that can bring them joy and"
            " laughter."
        ),
    ),
]

user_genre_documentary = [
    User(
        "Rachel Taylor",
        "F",
        23,
        (
            "an intellectually curious and socially conscious individual who has a deep"
            " appreciation for documentary films. With a thirst for knowledge and a"
            " desire to understand the world around her, she finds solace and"
            " inspiration in thought-provoking documentaries. Rachel consistently"
            " assigns high ratings (between 8-10) to documentary films because she"
            " believes in their power to inform, educate, and shed light on important"
            " social, cultural, and historical topics. She values the ability of"
            " documentaries to challenge perspectives and spark meaningful"
            " conversations, making them a vital medium for understanding the"
            " complexities of our world. However, when it comes to films outside the"
            " documentary genre, Rachel tends to feel disconnected and rates them low"
            " (between 1-5), as she strongly prefers movies that offer informative and"
            " thought-provoking content."
        ),
    ),
    User(
        "Michael Young",
        "M",
        19,
        (
            "an analytical and insightful man, has an insatiable appetite for"
            " documentary films. He is fascinated by real-life stories and enjoys"
            " delving into the depth and authenticity that documentaries offer."
            " Michael's high ratings (between 8-10) for documentary films stem from his"
            " admiration for the genre's ability to explore diverse subjects with"
            " nuance and authenticity. He believes that documentaries have the power to"
            " give voice to marginalized communities, shed light on pressing issues,"
            " and inspire positive change. Michael values the ability of documentaries"
            " to uncover untold stories and provide a window into different cultures,"
            " making them an invaluable source of knowledge and empathy. Conversely,"
            " movies outside the documentary genre receive a low rating (between 1-5)"
            " from Michael, as he tends to have less interest in fictional narratives."
        ),
    ),
    User(
        "Laura Jenkins",
        "F",
        65,
        (
            "a compassionate and empathetic woman who finds great resonance in"
            " documentary films. She is deeply moved by the human stories, struggles,"
            " and triumphs depicted in these films. Laura consistently assigns high"
            " ratings (between 8-10) to documentary films because she believes in their"
            " ability to foster empathy and ignite social consciousness. She sees"
            " documentaries as a powerful tool for creating awareness and driving"
            " meaningful change. Through their honest and raw storytelling,"
            " documentaries inspire Laura to take action, making them an integral part"
            " of her cinematic preferences. However, when it comes to non-documentary"
            " films, Laura tends to find them less impactful and rates them low"
            " (between 1-5), as her passion lies in the real stories and issues that"
            " documentaries explore."
        ),
    ),
    User(
        "David Giu",
        "M",
        73,
        (
            "an inquisitive and open-minded individual, gravitates towards documentary"
            " films as a means of expanding his horizons. He enjoys exploring different"
            " cultures, historical events, and scientific discoveries through the lens"
            " of real-life stories. David consistently gives high ratings (between"
            " 8-10) to documentary films because he appreciates their ability to"
            " enlighten, educate, and challenge preconceived notions. He believes that"
            " documentaries offer a unique blend of entertainment and education,"
            " allowing viewers to engage with important topics on a deeper level. David"
            " sees documentaries as a gateway to understanding the world around him and"
            " values their ability to stimulate curiosity and critical thinking."
            " However, for films that do not fall within the documentary genre, David"
            " generally assigns a low rating (between 1-5), as they often fail to"
            " capture his interest and provide the same level of educational value."
        ),
    ),
]
user_genre_fantasy = [
    User(
        "Emily Peterson",
        "F",
        23,
        (
            "a dreamer and an avid lover of the fantasy genre. She finds solace in the"
            " magical worlds, mythical creatures, and epic adventures depicted in"
            " fantasy films. Emily consistently assigns high ratings (between 8-10) to"
            " fantasy movies because she is captivated by their ability to transport"
            " her to enchanting realms and ignite her imagination. She loves exploring"
            " fantastical settings, encountering extraordinary characters, and"
            " witnessing epic battles between good and evil. Fantasy films provide an"
            " escape from reality and offer a sense of wonder and awe that deeply"
            " resonates with Emily. However, when it comes to films outside the fantasy"
            " genre, she tends to feel less engaged and rates them low (between 1-5),"
            " as they fail to capture her imagination in the same way."
        ),
    ),
    User(
        "Ethan Simmons",
        "M",
        29,
        (
            "a creative and imaginative individual who has a profound appreciation for"
            " the fantasy genre. He thrives on the fantastical elements, magical"
            " powers, and extraordinary quests found in fantasy films. Ethan"
            " consistently gives high ratings (between 8-10) to fantasy movies because"
            " he is fascinated by their ability to transport him to imaginative worlds"
            " and challenge the boundaries of reality. He appreciates the escapism and"
            " the sense of adventure that fantasy films provide. For Ethan, the genre"
            " offers a space where anything is possible, and he revels in the wonder"
            " and excitement it brings. However, films outside the fantasy genre"
            " receive a low rating (between 1-5) from Ethan, as they often lack the"
            " imaginative elements that he finds so captivating."
        ),
    ),
    User(
        "Olivia Nelson",
        "F",
        58,
        (
            "a whimsical and imaginative person who is deeply drawn to the fantasy"
            " genre. She finds joy in the magical and fantastical elements present in"
            " fantasy films. Olivia consistently assigns high ratings (between 8-10) to"
            " fantasy movies because she believes in their power to ignite the"
            " imagination, evoke a sense of wonder, and transport her to extraordinary"
            " realms. She appreciates the themes of heroism, friendship, and personal"
            " growth often found in fantasy stories. Olivia sees fantasy films as a"
            " source of inspiration and a way to explore themes of self-discovery and"
            " overcoming obstacles. When it comes to movies outside the fantasy genre,"
            " Olivia tends to rate them low (between 1-5), as they do not provide the"
            " same whimsical and imaginative experience that she seeks."
        ),
    ),
    User(
        "Liam Grayson",
        "M",
        58,
        (
            "an adventurous and imaginative soul who finds himself deeply immersed in"
            " the fantasy genre. He loves the magical realms, epic quests, and"
            " larger-than-life characters that fantasy films offer. Liam consistently"
            " assigns high ratings (between 8-10) to fantasy movies because he enjoys"
            " the sense of escapism and the opportunity to experience grand adventures."
            " He appreciates the rich world-building, imaginative storytelling, and the"
            " exploration of universal themes often found in fantasy films. However,"
            " for movies that do not fall within the fantasy genre, Liam tends to have"
            " less enthusiasm and rates them low (between 1-5), as they fail to capture"
            " his sense of wonder and excitement."
        ),
    ),
]
user_genre_romance = [
    User(
        "Sophia Collins",
        "F",
        21,
        (
            "a hopeless romantic who is deeply captivated by the romance genre. She"
            " finds solace in the heartfelt stories, emotional connections, and"
            " sweeping love affairs depicted in romance films. Sophia consistently"
            " assigns high ratings (between 8-10) to romance movies because she"
            " believes in their ability to evoke powerful emotions and transport her"
            " into a world of love and passion. She appreciates the chemistry between"
            " the characters, the tender moments, and the exploration of themes such as"
            " love, sacrifice, and personal growth. For Sophia, romance films offer an"
            " escape into a realm of heartwarming and enchanting love stories. However,"
            " when it comes to films outside the romance genre, she tends to feel less"
            " engaged and rates them low (between 1-5), as they fail to ignite the same"
            " emotional connection."
        ),
    ),
    User(
        "Jacob Walker",
        "M",
        18,
        (
            "a sentimental and affectionate individual who has a deep appreciation for"
            " the romance genre. He finds himself drawn to the enchanting tales of"
            " love, relationships, and the complexities of the human heart depicted in"
            " romance films. Jacob consistently gives high ratings (between 8-10) to"
            " romance movies because he believes in their ability to touch his soul and"
            " remind him of the power of love. He appreciates the emotional depth, the"
            " heartfelt performances, and the way romance films can explore the"
            " complexities of human relationships. For Jacob, romance movies have a"
            " special place in his heart, as they can evoke a range of emotions and"
            " leave a lasting impact. Conversely, movies outside the romance genre"
            " receive a low rating (between 1-5) from Jacob, as they often fail to"
            " resonate with him on a deep emotional level."
        ),
    ),
    User(
        "Ava Edwards",
        "F",
        55,
        (
            "a dreamy and idealistic person who is enamored by the romance genre. She"
            " finds joy in the enchanting love stories, the swoon-worthy moments, and"
            " the possibility of happily ever afters depicted in romance films. Ava"
            " consistently assigns high ratings (between 8-10) to romance movies"
            " because she believes in their ability to ignite her imagination and"
            " transport her into a world of passion and romance. She appreciates the"
            " emotional connections between the characters, the chemistry that unfolds,"
            " and the way romance films can evoke a sense of longing and anticipation."
            " For Ava, romance movies offer an escape into a realm of love and"
            " possibility. However, when it comes to non-romantic films, Ava tends to"
            " rate them low (between 1-5), as they do not provide the same emotional"
            " connection and enchantment that she seeks."
        ),
    ),
    User(
        "Oliver Ellis",
        "M",
        43,
        (
            "a sentimental and sensitive individual who has a profound appreciation for"
            " the romance genre. He finds solace in the emotional depth, the heartfelt"
            " storytelling, and the exploration of love and relationships depicted in"
            " romance films. Oliver consistently gives high ratings (between 8-10) to"
            " romance movies because he believes in their ability to stir his emotions"
            " and create a sense of connection. He appreciates the way romance films"
            " can convey universal themes of love, heartbreak, and personal growth. For"
            " Oliver, romance movies offer a space for introspection and an opportunity"
            " to explore the complexities of human emotions. Conversely, movies outside"
            " the romance genre receive a low rating (between 1-5) from Oliver, as they"
            " often fail to evoke the same emotional resonance and introspection."
        ),
    ),
]
user_genre_horror = [
    User(
        "Sarah Bailey",
        "F",
        32,
        (
            "an adrenaline junkie who thrives on the thrill and excitement that the"
            " horror genre provides. She is fascinated by the suspense, the jump"
            " scares, and the eerie atmosphere that horror films offer. Sarah"
            " consistently assigns high ratings (between 8-10) to horror movies because"
            " she enjoys the adrenaline rush and the sense of fear they evoke. She"
            " appreciates the creative storytelling, the clever plot twists, and the"
            " way horror films can tap into our deepest fears. For Sarah, horror movies"
            " offer an exhilarating experience that keeps her on the edge of her seat."
            " However, when it comes to films outside the horror genre, she tends to"
            " feel less engaged and rates them low (between 1-5), as they fail to"
            " provide the same level of excitement and suspense."
        ),
    ),
    User(
        "James Clarke",
        "M",
        24,
        (
            "a curious and daring individual who is drawn to the horror genre. He finds"
            " himself fascinated by the psychological exploration, the supernatural"
            " elements, and the spine-chilling narratives depicted in horror films."
            " James consistently gives high ratings (between 8-10) to horror movies"
            " because he appreciates their ability to delve into the darker aspects of"
            " human nature and explore our deepest fears. He enjoys the thrill of being"
            " scared and the way horror films can challenge our perceptions. For James,"
            " horror movies offer a unique experience that allows him to confront his"
            " own fears in a controlled environment. Conversely, films outside the"
            " horror genre receive a low rating (between 1-5) from James, as they often"
            " lack the suspense and intensity that he seeks."
        ),
    ),
    User(
        "Emma Dixon",
        "F",
        41,
        (
            "a thrill-seeker who finds herself drawn to the horror genre. She enjoys"
            " the adrenaline rush, the chilling atmosphere, and the supernatural"
            " elements present in horror films. Emma consistently assigns high ratings"
            " (between 8-10) to horror movies because she relishes the feeling of being"
            " scared and the sense of anticipation they bring. She appreciates the"
            " clever storytelling, the suspenseful build-up, and the way horror films"
            " can keep her on the edge of her seat. For Emma, horror movies offer a"
            " thrilling escape from reality and a chance to experience intense emotions"
            " in a controlled setting. However, when it comes to films outside the"
            " horror genre, Emma tends to rate them low (between 1-5), as they fail to"
            " provide the same level of excitement and suspense."
        ),
    ),
    User(
        "Daniel Freeman",
        "M",
        56,
        (
            "a curious and fearless individual who finds himself fascinated by the"
            " horror genre. He enjoys exploring the supernatural, the paranormal, and"
            " the psychological aspects depicted in horror films. Daniel consistently"
            " gives high ratings (between 8-10) to horror movies because he appreciates"
            " their ability to evoke a sense of unease and tap into our primal fears."
            " He finds the tension and suspense in horror films exhilarating, and he"
            " enjoys the creative ways in which they can push the boundaries of"
            " storytelling. For Daniel, horror movies offer an escape into the unknown"
            " and a chance to experience intense emotions. Conversely, films outside"
            " the horror genre receive a low rating (between 1-5) from Daniel, as they"
            " often lack the thrilling elements and the ability to challenge his"
            " senses."
        ),
    ),
]
user_genre_family = [
    User(
        "Emily Griffiths",
        "F",
        16,
        (
            "warm-hearted and family-oriented individual who has a deep appreciation"
            " for the family genre. She finds joy in heartwarming stories, relatable"
            " characters, and the strong bonds depicted in family films. Emily"
            " consistently assigns high ratings (between 8-10) to family movies because"
            " she believes in their ability to bring people together and instill values"
            " of love, unity, and friendship. She appreciates the positive messages,"
            " the humor, and the heartwarming moments that family films often offer."
            " For Emily, family movies provide a sense of nostalgia and an opportunity"
            " to share meaningful experiences with loved ones. However, when it comes"
            " to films outside the family genre, she tends to feel less connected and"
            " rates them low (between 1-5), as they may not resonate with her values of"
            " togetherness and love."
        ),
    ),
    User(
        "Ethan Hayes",
        "M",
        28,
        (
            "a compassionate and caring individual who finds himself drawn to the"
            " family genre. He values the themes of love, loyalty, and personal growth"
            " often depicted in family films. Ethan consistently gives high ratings"
            " (between 8-10) to family movies because he appreciates their ability to"
            " evoke genuine emotions and explore the dynamics of relationships. He"
            " enjoys the heartwarming moments, the relatable characters, and the"
            " positive lessons that family films can impart. For Ethan, family movies"
            " offer a sense of comfort and a reminder of the importance of cherishing"
            " the bonds with our loved ones. Conversely, films outside the family genre"
            " receive a low rating (between 1-5) from Ethan, as they may not capture"
            " the same sense of connection and emotional resonance."
        ),
    ),
    User(
        "Olivia Ingram",
        "F",
        45,
        (
            "a nurturing and kind-hearted person who finds solace in the family genre."
            " She is drawn to the wholesome stories, the uplifting messages, and the"
            " strong values portrayed in family films. Olivia consistently assigns high"
            " ratings (between 8-10) to family movies because she believes in their"
            " ability to inspire, teach important life lessons, and promote empathy and"
            " understanding. She appreciates the positive role models, the heartfelt"
            " moments, and the sense of unity that family films often portray. For"
            " Olivia, family movies offer a source of comfort and a reminder of the"
            " power of love and compassion. However, when it comes to non-family films,"
            " Olivia tends to rate them low (between 1-5), as they may not provide the"
            " same level of warmth and meaningful storytelling."
        ),
    ),
    User(
        "Oliver v",
        "M",
        82,
        (
            "a caring and empathetic individual who resonates deeply with the family"
            " genre. He finds himself drawn to stories that explore the dynamics of"
            " familial relationships, the importance of acceptance, and the power of"
            " love. Oliver consistently assigns high ratings (between 8-10) to family"
            " movies because he appreciates their ability to evoke genuine emotions and"
            " touch his heart. He enjoys the heartwarming moments, the relatable"
            " characters, and the positive values that family films often highlight."
            " For Oliver, family movies offer a sense of comfort and a reminder of the"
            " significance of family bonds. Conversely, films outside the family genre"
            " receive a low rating (between 1-5) from Oliver, as they may not capture"
            " the same sense of connection and emotional resonance that he seeks."
        ),
    ),
]


class GenrePreferencePaperStudy(AbstractCaseStudy):
    name = "genre_preference_paper"

    def __init__(self, create_env, run_name, max_genres=8) -> None:
        super().__init__(create_env=create_env, run_name=run_name)
        self.max_genres = max_genres

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
            "Action": user_genre_action,
            "Animation": user_genre_animation,
            "Comedy": user_genre_comedy,
            "Documentary": user_genre_documentary,
            "Family": user_genre_family,
            "Fantasy": user_genre_fantasy,
            "Horror": user_genre_horror,
            "Romance": user_genre_romance,
        }
        figs = []
        ps_no = []
        ps_yes = []
        html_interactions = []

        data_no_acc = []
        data_yes_acc = []
        for genre, users in tqdm.tqdm(list(configs.items())[: self.max_genres]):
            # Set environment
            genre_lower = genre.lower()
            env_yes = self._get_env(
                users,
                f"../datasets/genres_20/{genre_lower}.json",
            )

            out_of_dist_yes = lambda x: x <= 7
            data_yes, vote_average_tmdb_yes = interact_sequential(
                env_yes, out_of_dist_yes
            )
            data_yes_acc.append(data_yes)
            env_no = self._get_env(
                users, f"../datasets/genres_20/no_{genre_lower}.json"
            )
            out_of_dist_no = lambda x: x >= 6
            data_no, vote_average_tmdb_no = interact_sequential(env_no, out_of_dist_no)
            data_no_acc.append(data_no)
            ratings_yes = data_to_matrix(env_yes, data_yes)
            ratings_no = data_to_matrix(env_no, data_no)
            fig_heatmaps = plot_heatmap_2_sides(
                ratings_yes,
                ratings_no,
                title=f"{genre}",
                subtitle1=f"Likes {genre} (Ratings)",
                subtitle2=f"No {genre} (Ratings)",
            )

            fig_Y_users = plot_users(ratings_yes, f"Likes {genre} (Users)")
            fig_N_users = plot_users(ratings_no, f"NO {genre} (Users)")

            fig_tmdb_yes = plot_tmdb_corr(
                ratings_yes, vote_average_tmdb_yes, f"{genre} (correlation TMDB)"
            )
            fig_tmdb_no = plot_tmdb_corr(
                ratings_no, vote_average_tmdb_no, f"No {genre} (correlation TMDB)"
            )

            figs.append(
                [
                    fig_heatmaps,
                    fig_Y_users,
                    fig_N_users,
                    fig_tmdb_yes,
                    fig_tmdb_no,
                ]
            )

            html_interactions.append(
                [
                    data_yes["LLM_interaction_HTML"].iloc[1],  # so we have history also
                    data_no["LLM_interaction_HTML"].iloc[1],  # so we have history also
                ]
            )

            perc_success_yes = (ratings_yes >= np.full_like(ratings_yes, 8)).mean()
            perc_success_no = (ratings_no <= np.full_like(ratings_no, 5)).mean()
            ps_no.append(ratings_no <= np.full_like(ratings_no, 5))
            ps_yes.append(ratings_yes >= np.full_like(ratings_yes, 8))

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
                "Genre": list(configs.keys())[: self.max_genres],
                "Percentage success": (ps_yes + ps_no) / 2,
                "Percentage success (Positive)": ps_yes,
                "Percentage success (Negative)": ps_no,
            }
        ).to_csv(f"{base_path}/{self.name}.csv", index=False)
