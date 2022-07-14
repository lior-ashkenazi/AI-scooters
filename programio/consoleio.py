from data.trafficdatatypes import *
from programio.abstractio import AbstractIO


class ConsoleIO(AbstractIO):

    @staticmethod
    def get_user_discrete_choice(prompt: str, choices: List[str]) -> str:
        print(prompt)
        print(f"legal values: {choices}")
        while True:
            choice: str = input("")
            if choice not in choices:
                print(f"legal values: {choices}")
                continue
            return choice

    @staticmethod
    def get_user_numerical_choice(prompt: str, low_bound: float, high_bound: float)\
            -> float:
        legal_range_msg = f"legal values: in range [{low_bound}, {high_bound}]"
        print(prompt)
        print(legal_range_msg)
        while True:
            choice = input("")
            try:
                choice = float(choice)
            except ValueError:
                print("please enter a numerical value")
                continue
            if choice < low_bound or choice > high_bound:
                print(legal_range_msg)
                continue
            return choice

    @staticmethod
    def show_spread(spread_points: List[NestAllocation]) -> None:
        print("scooters spread:")
        for nest_allocation in spread_points:
            if nest_allocation.scooters_num > 0:
                print(nest_allocation)

    @staticmethod
    def confirm_and_continue() -> None:
        input("Press enter to continue")

    @staticmethod
    def show_value(message: str, value) -> None:
        print(message)
        print(value)
