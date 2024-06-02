from environment.items_retrieval import ItemsRetrieval
from .book import Book
from abc import ABC, abstractmethod
import numpy as np
import typing
from environment.memory import UserMovieInteraction


class SimpleBookRetrieval(ItemsRetrieval):
    """
    Object that is responsable to retrieve items, the items are picked based on a simple similarity

    Attributes:
        num (integer): maximum number of items to retrieve
    """

    def __init__(self, num: int):
        self.num = num

    def similarity(self, item1: Book, item2: Book):
        """
        Similarity function with used to retrieve items, it construct a greedy similarity based on
        the authors, the categories, the vote average and the original language

        Args:
            item1 (Book): first item
            item2 (Book): second item

        Return:
            similarity score (float)
        """
        authors_id1 = item1.authors
        authors_id2 = item2.authors

        if len(authors_id1) > 0 and len(authors_id2) > 0:
            authors_intersection = [id for id in authors_id1 if id in authors_id2]
            authors_similarity = (
                2 * len(authors_intersection) / (len(authors_id1) + len(authors_id2))
            )
        elif len(authors_id1) == 0 and len(authors_id2) == 0:
            authors_similarity = 1
        else:
            authors_similarity = 0

        if len(item1.categories) > 0 and len(item2.categories) > 0:
            categories_intersection = [
                category
                for category in item1.categories
                if category in item2.categories
            ]
            categories_similarity = (
                2
                * len(categories_intersection)
                / (len(item1.categories) + len(item2.categories))
            )
        elif len(item1.categories) == 0 and len(item2.categories) == 0:
            categories_similarity = 1
        else:
            categories_similarity = 0

        vote_similarity = 1 - (abs(item1.vote_average - item2.vote_average) / 5)

        return np.mean(
            [
                categories_similarity,
                authors_similarity,
                vote_similarity,
            ]
        )

    def retrieve(
        self,
        curr_item: Book,
        item_list: typing.List[Book],
        interactions: typing.List[UserMovieInteraction],
    ) -> typing.Tuple[typing.List[Book], typing.List[UserMovieInteraction]]:
        """
        The retrieve function is responsable to retrieve most relevant items, in this case it is based on the similarity function.
        It sort the items in decreasing order based on the similarity with curr_item (Book) and picks the most relevant (num of them)
        Args:
            curr_item (Book): the item of interest
            item_list (List Book): a list of Books that the user has seen, from which we want to select the most similar to curr_item
            interactions (List Interaction): a list containing all the interaction of the user, the order in the list should correspond with th
                the order of item_list
        Return:
            retrieved_items (List Book): list containing the most relevant items
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
