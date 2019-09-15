from collections import defaultdict
from functools import total_ordering
from typing import List, Tuple

from base_solver import Solver
from n_armed_bandit import NArmedBandit


@total_ordering
class TrainResult:
    def __init__(self, bandit: NArmedBandit, solver: Solver, actions_selected: dict, rewards: List[float]):
        """
        Creates a new TrainResult.

        :param bandit: The bandit that was trained on
        :param solver: The solver that trained
        :param actions_selected: The actions which where selected during training
        :param rewards: A list containing all rewards gathered
        """
        self.bandit = bandit
        self.solver = solver
        self.actions_selected = actions_selected
        self.rewards = rewards
        self.reward_sum = sum(rewards)

    def __str__(self) -> str:
        """
        :return: Returns a string representation of this TrainResult
        """
        string_result = []
        bandit_optimal_action = self.bandit.get_optimal_action()
        solver_optimal_action = self.solver.get_greedy_action()

        string_result.append('action bandit solver optimal selections')
        for action in range(self.bandit.get_number_of_actions()):
            optimal_string = '{} {}'.format(
                'B' if action == bandit_optimal_action else ' ',
                'S' if action == solver_optimal_action else ' '
            )
            string_result.append('{action:^6d} {bandit:^ 6.3f} {solver:^ 6.3f} {optimal:^7} {selections:^ 10d}'.format(
                action=action,
                bandit=self.bandit.get_mean_rewards()[action],
                solver=self.solver.get_action_values()[action],
                optimal=optimal_string,
                selections=self.actions_selected[action]
            ))
        string_result.append('reward sum: {}'.format(self.reward_sum))

        return '\n'.join(string_result)

    def __lt__(self, other) -> bool:
        """
        Returns True if this training result has a lower reward sum than other

        :param other: The TrainResult to compare_once with
        :type other: TrainResult
        :return: Returns True if this training result has a lower reward sum than other, otherwise False
        """
        return self.reward_sum < other.reward_sum

    def __eq__(self, other) -> bool:
        """
        Returns True if this training result has the same reward sum than the other

        :param other: The TrainResult to compare_once with
        :type other: TrainResult
        :return: True, if this training result has the same reward sum than the other
        """
        return self.reward_sum == other.reward_sum


def train(bandit: NArmedBandit, solver: Solver, num_plays: int) -> TrainResult:
    """
    Trains the given solver on the given bandit.

    :param bandit: The bandit to train the solver on
    :param solver: The solver that tries to solve the given bandit
    :param num_plays: The number of plays during this training
    :return: The result of the training
    """
    rewards = []
    actions_selected = defaultdict(int)
    for i in range(num_plays):
        action, reward = solver.play(bandit)
        actions_selected[action] += 1
        rewards.append(reward)

    return TrainResult(bandit, solver, actions_selected, rewards)


def compare_once(bandit: NArmedBandit, solvers: List[Solver], num_plays: int) -> List[Tuple[Solver, TrainResult]]:
    """
    Compares the given solvers by training it once and returns a ranked list of the solvers together with their
    TrainResults.

    :param bandit: The bandit to train the solvers on
    :param solvers: The solvers to compare_once. This list should not be empty.
    :param num_plays: The number of times each solver is playing the bandit
    :return: The index of the best solver
    """
    assert solvers

    results = []

    for solver in solvers:
        result = train(bandit, solver, num_plays)
        results.append(result)

    solvers_and_results = zip(solvers, results)

    return sorted(solvers_and_results, key=lambda solver_and_result: solver_and_result[1], reverse=True)
