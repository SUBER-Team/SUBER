from abc import ABC, abstractmethod
from typing import Callable

import pandas as pd

from environment import Simulatio4RecSys
from environment.users import UsersLoader


class AbstractCaseStudy(ABC):
    name: str
    create_env: Callable[[str, UsersLoader], Simulatio4RecSys]

    def __init__(
        self,
        create_env: Callable[[str, UsersLoader], Simulatio4RecSys],
        run_name: str,
    ) -> None:
        super().__init__()
        self.create_env = create_env
        self.run_name = run_name

    @abstractmethod
    def run(self):
        pass
