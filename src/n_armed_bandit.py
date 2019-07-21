import numpy as np


class NArmedBandit:
    """
    A stationary n armed bandit, with n levels. Each level has a stationary mean reward, that is initially chooses from
    a standardised normal distribution.
    """
    def __init__(self, n: int):
        """
        Creates a new n armed bandit.

        :param n: The number of levels of this bandit
        """
        self._mean_rewards = np.random.normal(0, 1, n)

    def get_mean_rewards(self) -> np.ndarray:
        """
        :return: The mean rewards of the different actions.
        """
        return self._mean_rewards

    def play(self, action: int):
        """
        Tries the level with the index given by action. Returns the reward of this play.

        :param action: The index of the level to try
        :type action: int
        :return: The random reward generated by the chosen action
        :rtype: float
        """
        return np.random.normal(self._mean_rewards[action], 1)

    def get_optimal_action(self) -> int:
        """
        :return: The action index with the highest mean reward
        """
        return int(np.argmax(self._mean_rewards))

    def get_number_of_actions(self) -> int:
        return self._mean_rewards.shape[0]

    def __str__(self) -> str:
        result = ['{} armed bandit:\n'.format(self.get_number_of_actions())]

        optimal_action = self.get_optimal_action()

        for index, mean_reward in enumerate(self._mean_rewards):
            if index == optimal_action:
                result.append('-->{: }'.format(mean_reward))
            else:
                result.append('   {: }'.format(mean_reward))

        return '\n'.join(result)
