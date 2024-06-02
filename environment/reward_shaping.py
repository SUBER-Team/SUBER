import typing
from environment.movies.movie import Movie
from environment.memory import UserMovieInteraction
import numpy as np
from abc import ABC, abstractmethod
import math


class RewardShaping(ABC):
    """
    Object that is responsable to reshape the rewards

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
    def reshape(
        self,
        item_interactions: typing.List[UserMovieInteraction],
        rating: int,
    ) -> typing.Tuple[int, bool]:
        """
        reshapes the reward to the item, the main application is to fix some behaviour of the LLM, which tends
        for example to give always the same rating to the same item

        Args:
            item_interactions (list of UserMovieInteraction): the list of interaction of the current user with the current item
            rating (int): rating (based on the LLM answer)

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
        number = (math.floor(number / self.stepsize)) * self.stepsize
        if number < self.min_rating:
            return self.min_rating
        elif number > self.max_rating:
            return self.max_rating
        else:
            return number


class IdentityRewardShaping(RewardShaping):
    def __init__(self, stepsize=0.5, min_rating=1, max_rating=10, seed=42):
        super().__init__(stepsize, min_rating, max_rating, seed)

    def reshape(
        self, item_interactions: typing.List[UserMovieInteraction], rating: int
    ) -> float:
        return rating, False


class RewardReshapingExpDecayTime(RewardShaping):
    def __init__(self, q=0.1, stepsize=0.5, min_rating=1, max_rating=10, seed=42):
        self.q = q
        super().__init__(stepsize, min_rating, max_rating, seed)

    def reshape(
        self, item_interactions: typing.List[UserMovieInteraction], rating: int
    ) -> float:
        if len(item_interactions) == 1:
            return float(rating), False
        current_time = item_interactions[-1].timestamp
        last_time = item_interactions[-2].timestamp
        num_watches = item_interactions[-1].num_watches

        return (
            self.rating_fixing(
                float(rating)
                * math.pow(
                    self.q, float(num_watches) / (float(current_time - last_time))
                )
            ),
            False,
        )


class RewardReshapingRandomWatch(RewardShaping):
    def __init__(self, q=0.1, stepsize=0.5, min_rating=1, max_rating=10, seed=42):
        self.q = q
        super().__init__(stepsize, min_rating, max_rating, seed)

    def reshape(
        self, item_interactions: typing.List[UserMovieInteraction], rating: int
    ) -> float:
        if len(item_interactions) == 1:
            return float(rating), False
        current_time = item_interactions[-1].timestamp
        last_time = item_interactions[-2].timestamp
        num_watches = item_interactions[-1].num_watches

        p = math.pow(self.q, float(num_watches) / (float(current_time - last_time)))

        watch = bool(self.rng.choice([True, False], p=[p, 1 - p]))
        if watch:
            return float(rating), False
        else:
            return 0.0, False


class RewardReshapingTerminateIfSeen(RewardShaping):
    def __init__(self, q=0.1, stepsize=0.5, min_rating=1, max_rating=10, seed=42):
        self.q = q
        super().__init__(stepsize, min_rating, max_rating, seed)

    def reshape(
        self, item_interactions: typing.List[UserMovieInteraction], rating: int
    ) -> float:
        if len(item_interactions) == 1:
            return float(rating), False
        else:
            return 0.0, True
