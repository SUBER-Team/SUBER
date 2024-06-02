import typing
from environment.movies.movie import Movie
from environment.memory import UserMovieInteraction
from abc import ABC, abstractmethod
import numpy as np


class ItemsRetrieval(ABC):
    """
    Object that is responsable to retrieve items
    """

    @abstractmethod
    def retrieve(
        self,
        curr_item: Movie,
        item_list: typing.List[Movie],
        interactions: typing.List[UserMovieInteraction],
    ) -> typing.Tuple[typing.List[Movie], typing.List[UserMovieInteraction]]:
        """
        The retrieve function is responsable to select from a list of items, with corresponding ratings, timestamp and num watched (i.e. interactions),
        a subset of items that are more relevant in order to construct a prompt for the LLM.
        """
        pass


class SimpleMoviesRetrieval(ItemsRetrieval):
    """
    Object that is responsable to retrieve items, the items are picked based on a simple similarity

    Attributes:
        num (integer): maximum number of items to retrieve
    """

    def __init__(self, num: int):
        self.num = num

    def similarity(self, item1: Movie, item2: Movie):
        """
        Similarity function with used to retrieve items, it construct a greedy similarity based on
        the actors, the genres, the vote average and the original language

        Args:
            item1 (Movie): first item
            item2 (Movie): second item

        Return:
            similarity score (float)
        """
        actors_id1 = [actor.id for actor in item1.actors]
        actors_id2 = [actor.id for actor in item2.actors]

        if len(actors_id1) > 0 and len(actors_id2) > 0:
            actors_intersection = [id for id in actors_id1 if id in actors_id2]
            actors_similarity = (
                2 * len(actors_intersection) / (len(actors_id1) + len(actors_id2))
            )
        elif len(actors_id1) == 0 and len(actors_id2) == 0:
            actors_similarity = 1
        else:
            actors_similarity = 0

        if len(item1.genres) > 0 and len(item2.genres) > 0:
            genres_intersection = [
                genre for genre in item1.genres if genre in item2.genres
            ]
            genres_similarity = (
                2 * len(genres_intersection) / (len(item1.genres) + len(item2.genres))
            )
        elif len(item1.genres) == 0 and len(item2.genres) == 0:
            genres_similarity = 1
        else:
            genres_similarity = 0

        vote_similarity = 1 - (abs(item1.vote_average - item2.vote_average) / 10)

        if item1.director == item2.director:
            director_similarity = 1
        else:
            director_similarity = 0

        return np.mean(
            [genres_similarity, actors_similarity, vote_similarity, director_similarity]
        )

    def retrieve(
        self,
        curr_item: Movie,
        item_list: typing.List[Movie],
        interactions: typing.List[UserMovieInteraction],
    ) -> typing.Tuple[typing.List[Movie], typing.List[UserMovieInteraction]]:
        """
        The retrieve function is responsable to retrieve most relevant items, in this case it is based on the similarity function.
        It sort the items in decreasing order based on the similarity with curr_item (Movie) and picks the most relevant (num of them)
        Args:
            curr_item (Movie): the item of interest
            item_list (List Movie): a list of Movies that the user has seen, from which we want to select the most similar to curr_item
            interactions (List Interaction): a list containing all the interaction of the user, the order in the list should correspond with th
                the order of item_list
        Return:
            retrieved_items (List Movie): list containing the most relevant items
            retrieved_interactions (List Interaction): lis containing interactions corresponding to the items in retrieved_items
        """
        tmp_list = []
        for item, interaction in zip(item_list, interactions):
            tmp_list.append((self.similarity(item, curr_item), item, interaction))

        tmp_list.sort(key=lambda x: x[0], reverse=True)

        retrived_items = []
        retrieved_interactions = []

        for i, (similarity, item, interaction) in enumerate(tmp_list):
            if i >= self.num:
                break
            retrived_items.append(item)
            retrieved_interactions.append(interaction)

        return retrived_items, retrieved_interactions


class TimeItemsRetrieval(ItemsRetrieval):
    """
    Object responsable to retrieve items, the items are retrieved are always the most recent ones.

    Attributes:
        num (integer): maximum number of items to retrieve
    """

    def __init__(self, num: int):
        self.num = num

    def retrieve(
        self,
        curr_item: Movie,
        item_list: typing.List[Movie],
        interactions: typing.List[UserMovieInteraction],
    ) -> typing.Tuple[typing.List[Movie], typing.List[UserMovieInteraction]]:
        '''
        The retrieve function is responsable to retrieve most relevant items, in this case it is based on the time.
        It sort the items in decreasing order based on the time  and picks the most relevant (num of them)
        Args:
            curr_item (Movie): the item of interest
            item_list (List Movie): a list of Movies that the user has seen, from which we want to select the most recent ones
            interactions (List Interaction): a list containing all the interaction of the user, the order in the list should correspond with th
                the order of item_list
        Return:
            retrieved_items (List Movie): list containing the most recent items
            retrieved_interactions (List Interaction): lis containing interactions corresponding to the items in retrieved_items
        """
        '''
        tmp_list = list(zip(item_list, interactions))
        tmp_list.sort(key=lambda x: x[1].timestamp, reverse=True)

        retrived_items = []
        retrieved_interactions = []

        for i, (item, interaction) in enumerate(tmp_list):
            if i >= self.num:
                break
            retrived_items.append(item)
            retrieved_interactions.append(interaction)

        return retrived_items, retrieved_interactions


class BestWorstItemsRetrieval(ItemsRetrieval):
    """
    This is a special retrieval since it only returns the two films that re considere the best and the worst from the user.
    """

    def __init__(self):
        pass

    def retrieve(
        self,
        curr_item: Movie,
        item_list: typing.List[Movie],
        interactions: typing.List[UserMovieInteraction],
    ) -> typing.Tuple[typing.List[Movie], typing.List[UserMovieInteraction]]:
        """
        Function that returns the best item with higher score, and the item with lowest score (only the seen items). In this case
        the only two items retrieved are the ones considered the worst and the best from the user.

        Args:
            item_list (list Movie): list of item seen
            scores (list of integer): list of the scores of the items in item_list

        Return
            List of Movies containing:
                items_scores[max_idx][0] (Movie): item with the highest score
                items_scores[min_idx][0] (Movie): item with the lowest score
            List of Interactions containing:
                items_scores[max_idx][1] (integer): interaction (score, time) of best item
                items_scores[min_idx][1] (integer): interaction (score, time) of worst item
        """
        items_interactions = [
            (m, i) for (m, i) in zip(item_list, interactions) if i.score > 0
        ]
        if len(items_interactions) == 0:
            return None, None, None, None

        max_idx = 0
        min_idx = 0
        for i, (m, s) in enumerate(items_interactions):
            max_idx = i if s > items_interactions[max_idx][1].score else max_idx
            min_idx = i if s < items_interactions[min_idx][1].score else min_idx

        return (
            [items_interactions[max_idx][0], items_interactions[min_idx][0]],
            [items_interactions[max_idx][1], items_interactions[min_idx][1]],
        )


class SentenceSimilarityItemsRetrieval(ItemsRetrieval):
    def __init__(self, num: int, name_field_embedding: str) -> None:
        self.num = num
        self.name_field_embedding = name_field_embedding
        super().__init__()

    def retrieve(
        self,
        curr_item: Movie,
        item_list: typing.List[Movie],
        interactions: typing.List[UserMovieInteraction],
    ) -> typing.Tuple[typing.List[Movie], typing.List[UserMovieInteraction]]:
        """
        The retrieve function is responsable to select from a list of items, with corresponding ratings, timestamp and num watched (i.e. interactions),
        a subset of items that are more relevant in order to construct a prompt for the LLM.
        """

        def f(x: Movie):
            # cosine similarity
            c = np.array(curr_item.__getattribute__(self.name_field_embedding))
            x = np.array(x.__getattribute__(self.name_field_embedding))
            return np.dot(c, x) / (np.linalg.norm(c) * np.linalg.norm(x))

        item_interactions = list(zip(item_list, interactions))
        item_interactions.sort(key=lambda x: f(x[0]), reverse=True)

        return (
            [item for item, _ in item_interactions[: self.num]],
            [interaction for _, interaction in item_interactions[: self.num]],
        )
