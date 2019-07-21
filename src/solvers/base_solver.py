import numpy as np
from abc import ABC
from typing import Callable, Tuple

from n_armed_bandit import NArmedBandit


class Solver(ABC):
    def __init__(self, n):
        """
        Creates a solver for a n armed bandit problem.

        :param n: The number of actions to try
        """
        self._action_values = np.zeros(n, dtype=np.float)

    def get_number_of_actions(self):
        return self._action_values.shape[0]

    def get_action_values(self) -> np.ndarray:
        """
        :return: the action values of this solver
        """
        return self._action_values

    def _apply_action_reward(self, action_chosen, reward):
        """
        Applies the given reward for the action, that was chosen.

        :param action_chosen: The action, that was chosen.
        :type action_chosen: int
        :param reward: The reward that was achieved by playing the action chosen.
        :type reward: float
        """
        raise NotImplementedError('abstract function')

    def _choose_action(self):
        """
        :return: Returns an action to play on an n armed bandit.
        """
        raise NotImplementedError('abstract function')

    def get_greedy_action(self) -> int:
        """
        :return: the action with the highest action_value
        """
        return int(np.argmax(self._action_values))

    def get_random_action(self) -> int:
        """
        :return: A random action
        """
        return np.random.randint(self.get_number_of_actions())

    def play(self, bandit: NArmedBandit) -> Tuple[int, float]:
        """
        Chooses an action and tries it on the given n armed bandit. Afterwards applies the result to the action values.

        :param bandit: The n armed bandit to try an action on
        :return: A tuple (action_played, reward)
        """
        action = self._choose_action()
        reward = bandit.play(action)
        self._apply_action_reward(action, reward)
        return action, reward


class ActionUpdateSolver(Solver, ABC):
    """
    Implements the _choose_action function, to update the chosen action with the given update rule.
    """
    def __init__(self, n: int, update_rule: Callable[[float, float, int], float]):
        """
        Creates a new ActionUpdateSolver with the given update rule.

        :param n: The number of actions
        :type n: int
        :param update_rule: A callable of the form:
                            update_rule(current_action_value, reward, num_action_tries) -> new_action_value

                            - current_action_value is the current value of the chosen action
                            - reward is the reward achieved by choosing this action
                            - num_action_tries is the number, how often this action was tried before this play
        """
        super().__init__(n)

        self._number_action_tries = np.zeros(shape=n, dtype=np.int)
        self._update_rule = update_rule

    def play(self, bandit: NArmedBandit) -> Tuple[int, float]:
        """
        Chooses an action and tries it on the given n armed bandit. Afterwards applies the result to the action values.

        :param bandit: The n armed bandit to try an action on
        :return: A tuple (action_selected, reward)
        """
        action, reward = super().play(bandit)
        self._number_action_tries[action] += 1
        return action, reward

    def _apply_action_reward(self, action_chosen, reward):
        """
        Applies the given reward for the action, that was chosen with the given update_rule.

        :param action_chosen: The action, that was chosen.
        :type action_chosen: int
        :param reward: The reward that was achieved by playing the action chosen.
        :type reward: float
        """
        self._action_values[action_chosen] = self._update_rule(
            self._action_values[action_chosen],
            reward,
            self._number_action_tries[action_chosen]
        )
