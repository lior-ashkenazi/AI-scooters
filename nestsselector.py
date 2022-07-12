from io.consoleio import ConsoleIO
from agents.staticagent import StaticAgent


GET_PROBLEM_PROMPT = "Please type problem to solve from the options below:"

class NestsSelector:
    DYNAMIC = "dynamic"
    STATIC = "static"

    def __init__(self):
        self.io = ConsoleIO()  # todo - replace with factory for AbstractIO (user chooses which type)

    def single_run(self):
        self._get_general_data()
        problem = self.io.get_user_discrete_choice(GET_PROBLEM_PROMPT,
                                                   [NestsSelector.DYNAMIC,
                                                    NestsSelector.STATIC])
        match problem:
            case NestsSelector.STATIC:
                self.run_static_problem()
            case NestsSelector.DYNAMIC:
                self.run_dynamic_problem()

    def _get_general_data(self):
        """
        gets from the user the following things:
        - traffic data
        - number of optional nests
        - factor for incomes (incomes vs expenses tradeoff)
        - time to run
        :return:
        """
        pass

    def run_static_problem(self):
        pass

    def run_dynamic_problem(self):
        pass



if __name__ == '__main__':
    pass