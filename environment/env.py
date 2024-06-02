import os
import string
from functools import reduce

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces

from environment.item import ItemsLoader
from environment.items_retrieval import ItemsRetrieval
from environment.items_selection import ItemsSelector
from environment.LLM import LLMRater
from environment.memory import Memory
from environment.reward_perturbator import RewardPerturbator
from environment.reward_shaping import RewardShaping
from environment.users import UsersLoader


class Simulatio4RecSys(gym.Env):
    def __init__(
        self,
        render_mode: str,
        items_loader: ItemsLoader,
        users_loader: UsersLoader,
        items_selector: ItemsSelector,
        reward_perturbator: RewardPerturbator,
        items_retrieval: ItemsRetrieval,
        reward_shaping: RewardShaping,
        llm_rater: LLMRater,
        render_path: str = "./tmp/render/",
        evaluation: bool = False,
    ):
        """
        Initialize render mode, if render_mode == 'human', then at every step the console will print
        human readable information about the state and action
        """
        self.render_mode = render_mode
        self.render_path = render_path
        self.metadata = {"render_modes": ["human", "csv"]}

        """
        Initialize the users list and
        count how many users there are in the dataset
        """
        self.users_loader = users_loader
        self.user_list = self.users_loader.get_users()
        self.num_users = len(self.user_list)

        """
        Initialize the item loader
        """
        self.items_loader = items_loader
        self.item_ids = self.items_loader.load_all_ids()
        self.num_items = len(self.item_ids)

        """
        user_id (integer) between 0 and num_users
        user_name (string)
        user_gender ('M' or 'F'), represented by a discrete space of two elements
        user_age (integer) age of the user between 0 and 200
        user_description (string ); max lenght 10e4
        items_interact (list of tuple of integers), is a list of films ids seen and the rating assigned by the user (corresponding to user_id)

        """
        self.observation_space = spaces.Dict(
            {
                "user_id": spaces.Discrete(self.num_users),
                "user_name": spaces.Text(
                    max_length=100, min_length=1, charset=string.printable
                ),
                "user_gender": spaces.Discrete(2),
                "user_age": spaces.Box(low=0, high=200, shape=(1,), dtype=np.int_),
                "user_description": spaces.Text(
                    max_length=10000, min_length=1, charset=string.printable
                ),
                "items_interact": spaces.Sequence(
                    spaces.Box(
                        low=np.array([0, 0]),
                        high=np.array([self.num_items, 11]),
                        shape=(2,),
                        dtype=np.int_,
                    )
                ),
            }
        )

        """
        We have one action for each Movie in the dataset, one action correspond to recommend one item
        """
        self.action_space = spaces.Discrete(self.num_items)

        """
        The following dictionary mabs abstract action to films ids, such that each action correspond to recommending 
        the film that correspond to the mapped id and viceversa
        """
        self.action_to_item = {}
        self.item_to_action = {}
        count = 0
        for id in self.item_ids:
            self.action_to_item[count] = id
            self.item_to_action[id] = count
            count += 1

        """
        Initialize Memory
        """
        self.memory = Memory(self.items_loader)

        self.items_retrieval = items_retrieval
        self.items_selector = items_selector
        self.reward_perturbator = reward_perturbator
        self.rating_prompt = llm_rater
        self.llm_seed = 0

        """
        Reward shaping
        """
        self.reward_shaping = reward_shaping

        self.evaluation = evaluation
        self.evaluation_previous_user_id = None
        self.evaluation_count = 0

    def _get_obs(self):
        gender = 0 if self._user.gender == "M" else 1
        return {
            "user_id": self._user.id,
            "user_name": self._user.name,
            "user_gender": gender,
            "user_age": np.array([self._user.age], dtype=np.int_),
            "user_description": self._user.description,
            "items_interact": self._items_interact,
        }

    def reset(self, seed=None, options=None, user_id=None):
        """
        The reset function resets the environment, this is done by selecting a new user
        the user selection is performed at random if the user_id input is None
        """
        super().reset(seed=seed)
        self.clean_memory()
        if seed is not None:
            self.items_selector.seed(seed)
            self.reward_perturbator.seed(seed)
            self.llm_seed = seed

        """
        Initialize a new user by picking a random user id between 0 and num_users if user_id is None
        """
        if user_id is None and not self.evaluation:
            user_id = self.np_random.integers(low=0, high=self.num_users)
        elif self.evaluation:
            if self.evaluation_previous_user_id is None:
                user_id = 0
                self.evaluation_count = 0
            else:
                user_id = self.evaluation_previous_user_id + 1
                self.evaluation_count = 0
        self._user = self.user_list[user_id]

        self._items_interact = tuple()

        observation = self._get_obs()
        info = {}

        return observation, info

    def step(self, action: int):
        """
        The step takes an action, which correspond to a film recommendation. We use the mapping
        action_to_item to map an action to its corresponding film.
        """
        item_id = self.action_to_item[action]

        """
        Use the MoviesLoader to load curr_item, which is a Movie object crresponding to the id item_id 
        """
        curr_item = self.items_loader.load_items_from_ids(id_list=[item_id])

        """
        We fetch from the memory all previous films seen by the user together with the 
        interaction that the user had with the items. The interaction is represented via an interaction object 
        that summarize all important informations.
        """
        past_items, past_interactions = self.memory.get_items_and_scores(self._user.id)
        num_interacted = self.memory.get_num_interaction(self._user.id, item_id)

        """
        The next step is to retrieve from the list of all items seen a smaller list of relevant items. The relevance from the Movie
        depends on the retrieved mode.
        """
        retrieved_items, retrieved_interactions = self.items_retrieval.retrieve(
            curr_item[0], past_items, past_interactions
        )

        """
        Given the user, the recommended item and the retieved item we construct a prompt for the LLM to predict the rating that
        the user would give to the recommended Movie.
        """
        with torch.random.fork_rng(["cuda:0"]):
            torch.manual_seed(self.llm_seed)
            rating, explanation, html_interaction = self.rating_prompt.query(
                self._user,
                curr_item[0],
                num_interacted,
                retrieved_interactions,
                retrieved_items,
            )
            self.llm_seed += 1

        """
        After collecting the explanation and the rating from the LLM the next step is to select the item 
        (but this only in the case more than one is recommended)
        """
        selected_items, selected_ratings = self.items_selector.select(
            curr_item, [rating]
        )

        """
        Add a small perturbation to the rating.
        """
        selected_items, selected_ratings = self.reward_perturbator.perturb(
            curr_item, [rating]
        )

        """
        The next step consists in updating the Memory by adding the recommended item to the list of film seen by the user.
        """
        selected_items_ids = []

        for m in selected_items:
            selected_items_ids.append(m.id)

        self.memory.update_memory(self._user.id, selected_items_ids, selected_ratings)

        """
        We also update the state by adding the recommended item to the list of film seen
        """
        self._items_interact = self._items_interact + (
            np.array(
                [self.item_to_action[selected_items_ids[0]], selected_ratings[0]],
                dtype=np.int_,
            ),
        )

        """
        Termination is modelled in a similar fashion to a geometric distribution: after every step the user with some small probability
        stops intercating with the environment 
        """
        terminated = self.np_random.choice([True, False], p=[0.025, 0.975])
        terminated = bool(terminated)
        observation = self._get_obs()
        reward = reduce(lambda x, y: x + y, selected_ratings)
        info = {
            "LLM_explanation": explanation,
            "LLM_rating": rating,
            "LLM_interaction_HTML": html_interaction,
        }

        item_interaction = self.memory.user_to_seen_films[self._user.id][item_id]
        reward, reward_shaping_termination = self.reward_shaping.reshape(
            item_interaction, reward
        )
        if reward_shaping_termination:
            terminated = True

        # Handles evaluation termination
        if self.evaluation:
            self.evaluation_previous_user_id = self._user.id
            self.evaluation_count += 1
            terminated = False
            if self.evaluation_count == 11:
                terminated = True

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "human":
            print(
                f"User: {self._user.name}, List of interacted items:"
                f" {self._items_interact[:-1]}"
                + (
                    f" Item proposed: {self._items_interact[-1][0]}, User reward:"
                    f" {self._items_interact[-1][1]}  "
                    if len(self._items_interact) > 0
                    else ""
                )
            )
        if len(self._items_interact) > 0 and self.render_mode == "csv":
            df = pd.DataFrame(
                {
                    "user_id": [self._user.id],
                    "user_name": [self._user.name],
                    "time": [len(self._items_interact)],
                    "movie_id": np.stack(list(self._items_interact))[-1, 0],
                    "rating": np.stack(list(self._items_interact))[-1, 1],
                },
            )

            if os.path.exists(self.render_path):
                df.to_csv(
                    self.render_path,
                    mode="a",
                    index=False,
                    header=False,
                )
            else:
                os.makedirs(os.path.dirname(self.render_path), exist_ok=True)
                df.to_csv(
                    self.render_path,
                    index=False,
                )

    def clean_memory(self):
        self.memory = Memory(self.items_loader)

    def delete_user_item(self, user_id: int, action: int):
        """
        The function is designed to remove a user item interaction from the memory,
        effectively simulating the act of forgetting that a particular user has watched a specific item.

        Args:
            user_id (integer): user from which we want to delete a item
            action (integer): action that correspond to the item we want to delete, note that is not the item_id but the action that correspond to
                              recommending the item.
        """
        item_id = self.action_to_item[action]
        self.memory.delete_user_item(user_id, item_id)

    def delete_last_user_item(self, user_id: int, action: int):
        """
        The function is designed to remove the last user item interaction from the memory,
        effectively simulating the act of forgetting that a particular user has watched a specific item last time.

        Args:
            user_id (integer): user from which we want to delete a item
            action (integer): action that correspond to the item we want to delete, note that is not the item_id but the action that correspond to
                              recommending the item.
        """
        item_id = self.action_to_item[action]
        self.memory.delete_last_user_item_interaction(user_id, item_id)
