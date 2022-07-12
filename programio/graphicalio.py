from abstractio import AbstractIO
from typing import List


class GraphicIO(AbstractIO):
    def __init__(self):
        pass

    @staticmethod
    def get_user_discrete_choice(prompt: str, choices: List[str]) -> str:
        pass

    @staticmethod
    def get_user_numerical_choice(prompt: str, low_bound, high_bound) -> float:
        pass