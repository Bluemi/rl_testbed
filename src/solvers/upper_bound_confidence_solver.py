from typing import Callable

import numpy as np

from base_solver import ActionUpdateSolver


class UpperBoundConfidenceSolver(ActionUpdateSolver):
    def __init__(
            self,
            name: str,
            n: int,
            update_rule: Callable[[float, float, int], float],
            initial_action_value: float = 0.0,
            confidence: float = 0.0
    ):
        """
        Creates a new UpperBoundConfidenceSolver.

        :param name: The name of this solver
        :param n: The number of actions
        :param update_rule: A callable of the form:
                            update_rule(current_action_value, reward, num_action_tries) -> new_action_value

                            - current_action_value is the current value of the chosen action
                            - reward is the reward achieved by choosing this action
                            - num_action_tries is the number, how often this action was tried before this play
        :param initial_action_value: The value every action value should start with
        :param confidence: A hyperparameter specifying how much actions should be selected, which are less explored
        """
        super().__init__(name, n, update_rule, initial_action_value)
        self._confidence = confidence

    def _choose_action(self):
        """
        Chooses an action by the upper confidence bound action selection.

        :return: The action to play following the upper confidence bound action selection.
        """
        ln_t = np.log(self.get_num_plays() + 1)
        confidence_values = self._confidence * np.sqrt(ln_t / self._number_action_tries)
        action = np.argmax(self._action_values + confidence_values)
        return action

