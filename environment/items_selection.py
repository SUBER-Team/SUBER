import typing
from environment.movies.movie import Movie
from environment.memory import UserMovieInteraction
import numpy as np
from abc import ABC, abstractmethod


class ItemsSelector(ABC):
    """
    Abstract object responsable to select Movies, given a list of items, with predicted rating (from LLM) selects the one that the user will see.

    Attributes:
        seed (integer): seed of the selector
    """

    def __init__(self, seed=42):
        self.seed(seed)

    @abstractmethod
    def select(
        self, items: typing.List[Movie], ratings: typing.List[float]
    ) -> typing.Tuple[typing.List[Movie], typing.List[float]]:
        """Function used to select items"""
        pass

    def seed(self, seed: int):
        """
        Function to change the seed of the selector

        Args:
            seed (integer): new seed of the selector
        """
        self.rng = np.random.default_rng(seed)


class GreedySelector(ItemsSelector):
    """
    Object that is responsable to select Movies, given a list of items, with predicted rating (from LLM) selects the one that the user will see.
    In this case it selects the one with higher predicted rating.
    This is used in case actions can correspond to recommending more than one film.

    Attributes:
        seed (integer): seed of the selector
    """

    def __init__(self, seed=42):
        super().__init__(seed)

    def select(
        self, items: typing.List[Movie], ratings: typing.List[float]
    ) -> typing.Tuple[typing.List[Movie], typing.List[float]]:
        """
        Select greedly one item based on the rating, in this case it makes a deterministic choice and picks the film
        with highest rating, in case there is more than one we pick the one with higher vote_average

        Args:
            items (list of Movie): the list of items the user has to select from
            ratings (int): the list of ratings the user would give (based on the LLM answer)

        Return
            a list of items that contains the same items as before (potentially permuted)
            a list with rating zero (= not watched this film) and only one film with rating > 0, and correspond to the film seen
        """
        items_ratings = list(zip(items, ratings))
        items_ratings.sort(key=lambda x: (x[1], x[0].vote_average), reverse=True)
        selected_items = [item for (item, rating) in items_ratings]
        selected_ratings = [0] * len(items)
        selected_ratings[0] = items_ratings[0][1]
        return selected_items, selected_ratings


class GreedySelectorRandom(ItemsSelector):
    """
    Object that is responsable to select Movies, given a list of items, with predicted rating (from LLM), selects the one that the user will see.
    In this case it selects the one with higher predicted rating, but with some small probabilities chooses that the user is not going to watch any film.
    This is used in case actions can correspond to recommending more than one film.

    Attributes:
        p (float): probabilities of the user to watch the recommended item
        seed (integer): seed of the selector
    """

    def __init__(self, p=0.9, seed=42):
        self.greedy_selector = GreedySelector(seed)
        super().__init__(seed)
        self.p = p

    def select(
        self, items: typing.List[Movie], ratings: typing.List[float]
    ) -> typing.Tuple[typing.List[Movie], typing.List[float]]:
        """
        Select greedly one item based on maximum rating, but with small probability chooses not to look at any film.

        Args:
            items (list of Movie): the list of items the user has to select from
            ratings (int): the list of ratings the user would give (based on the LLM answer)

        Return
            list o items and their ratings (0 if not seen)
        """
        look_item = self.rng.choice([True, False], p=[self.p, 1 - self.p])
        if look_item:
            return self.greedy_selector.select(items, ratings)
        return items, [0] * len(items)

    def seed(self, seed: int):
        super().seed(seed)
        self.greedy_selector.seed(seed)


class RandomSelector(ItemsSelector):
    """
    Object that is responsable to select Movies, given a list of items, with predicted rating (from LLM), selects the one that the user will see.
    In this case can select more tha one item. For every recommended item the user will watch it with probability p

    Attributes:
        p (float): probability to watch a item
        seed (integer): seed of the selector
    """

    def __init__(self, p=0.5, seed=42):
        super().__init__(seed)
        self.p = p

    def select(
        self, items: typing.List[Movie], ratings: typing.List[float]
    ) -> typing.Tuple[typing.List[Movie], typing.List[float]]:
        """
        Select film completely at random

        Args:
            items (list of Movie): the list of items the user has to select from
            ratings (int): the list of ratings the user would give (based on the LLM answer)

        Return
            list o items and their ratings (0 if not seen)
        """
        random_vector = self.rng.choice([0, 1], size=len(ratings), p=[0.5, 0.5])
        selected_ratings = [a * b for a, b in zip(items, ratings)]
        return items, selected_ratings
