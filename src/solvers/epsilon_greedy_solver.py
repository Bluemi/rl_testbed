from typing import Callable
import numpy as np

from solvers.base_solver import ActionUpdateSolver
from n_armed_bandit import Playable
from update_rules import sample_average_update_rule


class EpsilonGreedySolver(ActionUpdateSolver):
    def __init__(
            self,
            name: str,
            n: int,
            update_rule: Callable[[float, float, int], float],
            epsilon: float,
            initial_action_value: float = 0.0
    ):
        """
        Creates a new EpsilonGreedySolver which chooses the greedy action with the probability (1 - epsilon) and
        otherwise the greedy action.

        :param name: The name of this solver
        :param n: The number of actions
        :param update_rule: A callable of the form:
                            update_rule(current_action_value, reward, num_action_tries) -> new_action_value

                            - current_action_value is the current value of the chosen action
                            - reward is the reward achieved by choosing this action
                            - num_action_tries is the number, how often this action was tried before this play
        :param epsilon: The probability of choosing a non-greedy action
        :param initial_action_value: The value every action value should start with
        """
        super().__init__(
            name=name,
            n=n,
            update_rule=update_rule,
            initial_action_value=initial_action_value)
        self._epsilon = epsilon

    def _choose_action(self) -> int:
        """
        This chooses a non-greedy action with the probability epsilon and the greedy action otherwise.

        :return: Returns an action to play on an n armed bandit.
        """
        if np.random.ranf() < self._epsilon:  # non-greedy action
            return self.get_random_action()
        else:
            return self.get_greedy_action()


def test_action_update_solver():
    solver = EpsilonGreedySolver('solver', 1, sample_average_update_rule, 0.1)

    class MockBandit(Playable):
        def __init__(self):
            self.i = 0

        def play(self, action):
            self.i += 1
            return self.i

    bandit = MockBandit()
    num_plays = 5
    sum_i = 0
    for i in range(num_plays):
        solver.play(bandit)
        sum_i += bandit.i

    assert solver.get_action_values()[0] == (sum_i / num_plays)
