from typing import Callable
import numpy as np

from base_solver import ActionUpdateSolver


class EpsilonGreedySolver(ActionUpdateSolver):
    def __init__(self, n: int, update_rule: Callable[[float, float, int], float], epsilon: float):
        """
        Creates a new EpsilonGreedySolver which chooses the greedy action with the probability (1 - epsilon) and
        otherwise the greedy action.

        :param epsilon: The probability of choosing a non-greedy action
        """
        super().__init__(n, update_rule)
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
