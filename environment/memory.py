import time
import typing


class UserMovieInteraction:
    """
    Object used to store an interaction between an user and a item,
    it contains the rating, the timestamp and the number of times the user has watched the item until this interaction.
    """

    def __init__(self, rating: float, timestamp: int, num_watches: int):
        self.rating: float = rating
        self.timestamp: int = timestamp
        self.num_watches = num_watches


class Memory:
    """
    Storage of information: a mapping from item IDs to UserMovieInteraction for each user.
    This allows us to keep track of the list of items watched by the user, along with the corresponding scores provided by the user and
    others relevant information like the timestamp and number of time one watched a item.
    As convention we save in the memory for every user and item id a list of all interaction between the user and the item in cronological order.
    """

    user_to_seen_films: typing.Dict[
        int, typing.Dict[int, typing.List[UserMovieInteraction]]
    ]

    def __init__(self, items_loader):
        self.user_to_seen_films = {}
        self.user_num_items_interact = {}
        self.items_loader = items_loader

    def update_memory(
        self, user_id: int, items_ids: typing.List[int], scores: typing.List[float]
    ):
        """
        Updates the memory for a given user with respect to new item IDs and scores.

        Args:
            user_id (integer): id of a user
            items_ids (list of integers): list of all recommended items ids
            scores (list of integers): the corresponding scores to the items

        Return:
            None
        """
        if user_id not in self.user_to_seen_films:
            self._initialize_user(user_id)
        for i, item_id in enumerate(items_ids):
            self.user_num_items_interact[user_id] += 1
            if item_id in self.user_to_seen_films[user_id]:
                self.user_to_seen_films[user_id][item_id].append(
                    UserMovieInteraction(
                        scores[i],
                        self.user_num_items_interact[user_id],
                        len(self.user_to_seen_films[user_id][item_id]) + 1,
                    )
                )
            else:
                self.user_to_seen_films[user_id][item_id] = [
                    UserMovieInteraction(
                        scores[i], self.user_num_items_interact[user_id], 1
                    )
                ]

    def _initialize_user(self, user_id: int):
        """
        Initialize a user if not seen yet

        Args:
            user_id (integer): id of a user

        Return:
            None
        """
        self.user_to_seen_films[user_id] = {}
        self.user_num_items_interact[user_id] = 0

    def _get_items_ids_and_interactions(self, user_id: int):
        """
        For a given user compute the list of items seen/recommended with the correspective ratings,
        by convention we return the last interaction

        Args:
            user_id (integer): id of a user

        Return:
            list_items_ids (list of integers): ids recommended
            list_interactions (UserMovieInteraction): list of scores and timestamp corresponding to the items
        """
        list_items_ids = []
        list_interactions = []
        if user_id not in self.user_to_seen_films:
            self._initialize_user(user_id)
            return [], []
        for item_id, interaction in self.user_to_seen_films[user_id].items():
            list_items_ids.append(item_id)
            list_interactions.append(interaction[-1])
        return list_items_ids, list_interactions

    def get_items_and_scores(self, user_id: int):
        """
        Uses internal function _get_items_ids_and_scores to return a
        list of Movie object with correspective scores

        Args:
            user_id (integer): id of a user
        """
        list_items_ids, list_interactions = self._get_items_ids_and_interactions(
            user_id
        )
        return (
            self.items_loader.load_items_from_ids(list_items_ids),
            list_interactions,
        )

    def delete_user_item(self, user_id: int, item_id: int):
        """
        The function is designed to remove all user item interaction from the memory,
        effectively simulating the act of forgetting that a particular user has watched a specific item.

        Args:
            user_id (integer): user from which we want to delete a item
            item_id (integer): id of the item we want to delete
        """
        del self.user_to_seen_films[user_id][item_id]

    def delete_last_user_item_interaction(self, user_id: int, item_id: int):
        """
        The function is designed to remove the last user item interaction from the memory,
        effectively simulating the act of forgetting that a particular user has watched a specific item last time.

        Args:
            user_id (integer): user from which we want to delete a item
            item_id (integer): id of the item we want to delete
        """
        self.user_to_seen_films[user_id][item_id].pop()
        if self.user_to_seen_films[user_id][item_id] == []:
            self.delete_user_item(user_id, item_id)

    def get_num_interaction(self, user_id: int, item_id: int):
        """
        Return the number of times a user has watched a item

        Args:
            user_id (integer): user from which we want to delete a item
            item_id (integer): id of the item we want to delete
        """
        if int(item_id) not in self.user_to_seen_films[user_id]:
            return 0
        return self.user_to_seen_films[user_id][item_id][-1].num_watches
