import os
from typing import List
from .user import User
from abc import ABC, abstractmethod
import pandas as pd


class UsersLoader(ABC):
    @abstractmethod
    def get_users(self) -> List[User]:
        pass


class UsersListLoader(UsersLoader):
    def __init__(
        self,
        users_list: List[User],
    ):
        self.users_list = users_list

    def get_users(self) -> List[User]:
        i = 0
        for user in self.users_list:
            user.id = i
            i += 1
        return self.users_list


class UsersCSVLoader(UsersLoader):

    """
    This class is used to load users from a csv file.
    The csv file must have the following columns:
    - name
    - gender
    - age
    - description

    The default location of the csv is in environment/users/datasets

    Args:
        name (str): name of the csv file
        base_dir (str, optional): base directory of the csv file. Defaults to "./datasets".

    """

    def __init__(
        self,
        name: str,
        base_dir: str = os.path.join(os.path.dirname(__file__), "./datasets"),
    ):
        self.path = os.path.join(base_dir, name + ".csv")

    def get_users(self) -> List[User]:
        df = pd.read_csv(self.path)
        users_list = []
        i = 0
        for index, row in df.iterrows():
            user = User(
                name=row["name"],
                gender=row["gender"],
                age=int(row["age"]),
                description=row["description"],
                job=row["job"] if "job" in row else "",
                hobby=row["hobby"] if "hobby" in row else "",
            )
            user.id = i
            i += 1
            users_list.append(user)
        return users_list
