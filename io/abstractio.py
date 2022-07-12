from typing import List
from abc import abstractmethod


class AbstractIO:

    @staticmethod
    @abstractmethod
    def get_user_discrete_choice(prompt: str, choices: List[str]) -> str:
        pass

