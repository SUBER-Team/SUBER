import typing
from environment.movies.movie import Movie
from environment.memory import UserMovieInteraction
import numpy as np
from abc import ABC, abstractmethod


class RewardPerturbator(ABC):
    """
    Object that is responsable to perturbate the predicted reward of the LLM.

    Attributes:
        stepsize (float): stepsize describe how the ratings should be rounded after perturbation, for example is stepsize = 0.5 the possible
                          ratings will me {min_rating, min_rating + 0.5, min_rating + 1, ..., max_rating}
        min_rating (integer): smalles rating that can be assigned to a item
        max_rating (integer): largest rating that can be assigned to a item
        seed (integer): seed of the perturbator
    """

    def __init__(self, stepsize=0.5, min_rating=1, max_rating=10, seed=42):
        self.seed(seed)
        self.stepsize = stepsize
        self.min_rating = min_rating
        self.max_rating = max_rating

    @abstractmethod
    def perturb(
        self, items: typing.List[Movie], ratings: typing.List[int]
    ) -> typing.Tuple[typing.List[Movie], typing.List[float]]:
        """
        Perturbes a bit the ratings from the LLM

        Args:
            items (list of Movie): the list of items the user has to select from
            ratings (list of int): the list of ratings the user would give (based on the LLM answer)

        Return
            the same list of items and the perturbated ratings
        """
        pass

    def seed(self, seed: int):
        """
        Function used to change the seed of the perturbator

        Args:
            seed (integer): new seed of the object
        """
        self.rng = np.random.default_rng(seed)

    def rating_fixing(self, number: float):
        """
        Function used to project a score to the set of feasible ratings, from min_rating to max_rating, spaced linearly every 0.5

        Args:
            number (float), number to project into the set of feasible numbers

        Return:
            projected number
        """
        number = (round(number / self.stepsize)) * self.stepsize
        if number < self.min_rating:
            return self.min_rating
        elif number > self.max_rating:
            return self.max_rating
        else:
            return number


class GreedyPerturbator(RewardPerturbator):
    """
    Object that is responsable to perturbate the predicted reward of the LLM. It changes the ratings in two ways: for every rating either it removes one or it adds
    one.

    Attributes:
        p (List of integer of size 3): contains the 3 probability of the perturbation
            p[0]: probability to remove 1 from the rating
            p[1]: probability of keeping the rating unchanged
            p[2]: probability to add 1 to the rating

        stepsize (float): stepsize describe how the ratings should be rounded after perturbation, for example is stepsize = 0.5 the possible
                        ratings will me {min_rating, min_rating + 0.5, min_rating + 1, ..., max_rating}
        min_rating (integer): smalles rating that can be assigned to a item
        max_rating (integer): largest rating that can be assigned to a item
        seed (integer): seed of the perturbator
    """

    def __init__(
        self, p=[0.1, 0.8, 0.1], stepsize=0.5, min_rating=1, max_rating=10, seed=42
    ):
        super().__init__(stepsize, min_rating, max_rating, seed)
        self.p = p
        assert len(p) == 3

    def perturb(
        self, items: typing.List[Movie], ratings: typing.List[int]
    ) -> typing.Tuple[typing.List[Movie], typing.List[float]]:
        """
        Perturbes a bit the ratings from the LLM

        Args:
            items (list of Movie): the list of items the user has to select from
            ratings (int): the list of ratings the user would give (based on the LLM answer)

        Return
            the same list of items and the perturbated ratings
        """

        perturbation = self.rng.choice([-1, 0, 1], size=len(ratings), p=self.p)
        temp = list(zip(ratings, perturbation))
        final_ratings = list(
            map(
                lambda x: (
                    self.rating_fixing(x[0] + x[1])
                    if (x[0] > 0 and self.rating_fixing(x[0] + x[1]) > 0)
                    else x[0]
                ),
                temp,
            )
        )
        return items, final_ratings


class GaussianPerturbator(RewardPerturbator):
    """
    Object that is responsable to perturbate the predicted reward of the LLM. It changes the rating by adding some
    Gaussian noise with mean and std to the ratings.

    Attributes:
        mean (float): mean of the Gaussian distribution used to perturbate
        std (float): std of the Gaussian distribution used to perturbate

        stepsize (float): stepsize describe how the ratings should be rounded after perturbation, for example is stepsize = 0.5 the possible
                        ratings will me {min_rating, min_rating + 0.5, min_rating + 1, ..., max_rating}
        min_rating (integer): smalles rating that can be assigned to a item
        max_rating (integer): largest rating that can be assigned to a item
        seed (integer): seed of the perturbator
    """

    def __init__(
        self, mean=0, std=0.5, stepsize=0.5, min_rating=1, max_rating=10, seed=42
    ):
        super().__init__(stepsize, min_rating, max_rating, seed)
        self.mean = mean
        self.std = std

    def perturb(
        self, items: typing.List[Movie], ratings: typing.List[int]
    ) -> typing.Tuple[typing.List[Movie], typing.List[float]]:
        """
        Perturbate the ratings (only of the seen items) by adding a random Gaussina Noise N(0,0.5).
        We ensure that a rating above zero stays above zero

        Args:
            items (list of Movie): the list of items the user has to select from
            ratings (int): the list of ratings the user would give (based on the LLM answer)

        Return
            the same list of items and the perturbated ratings
        """
        perturbation = self.rng.normal(self.mean, self.std, len(ratings))
        temp = list(zip(ratings, perturbation))
        final_ratings = list(
            map(
                lambda x: (
                    self.rating_fixing(x[0] + x[1])
                    if (x[0] > 0 and self.rating_fixing(x[0] + x[1]) > 0)
                    else x[0]
                ),
                temp,
            )
        )
        return items, final_ratings


class NoPerturbator(RewardPerturbator):
    """
    This perturbator does not changes the ratings.
    """

    def __init__(self, stepsize=0.5, min_rating=1, max_rating=10, seed=42):
        super().__init__(stepsize, min_rating, max_rating, seed)

    def perturb(
        self, items: typing.List[Movie], ratings: typing.List[int]
    ) -> typing.Tuple[typing.List[Movie], typing.List[float]]:
        return items, ratings
