from typing import List
from abstractio import AbstractIO


class ConsoleIO(AbstractIO):
    def __init__(self):
        pass

    @staticmethod
    def get_user_discrete_choice(prompt: str, choices: List[str]) -> str:
        while True:
            print(prompt)
            print(choices)
            choice: str = input("")
            if choice not in choices:
                continue
            return choice
