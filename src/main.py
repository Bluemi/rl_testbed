from collections import defaultdict

from base_solver import Solver
from n_armed_bandit import NArmedBandit
from solvers.epsilon_greedy_solver import EpsilonGreedySolver
from update_rules import sample_average_update_rule


NUM_ACTIONS = 5


def main():
    bandit = NArmedBandit(NUM_ACTIONS)
    solver = EpsilonGreedySolver(NUM_ACTIONS, sample_average_update_rule, 0.1)
    sum_reward = 0
    actions_selected = defaultdict(lambda: 0)
    for i in range(100):
        action, reward = solver.play(bandit)
        actions_selected[action] += 1
        sum_reward += reward

    print_result(bandit, solver, actions_selected, sum_reward)


def print_result(bandit: NArmedBandit, solver: Solver, actions_selected: defaultdict, sum_reward: float):
    """
    Prints the result of the plays.

    :param bandit: The bandit with which was played
    :param solver: The solver that tried to solve the problem
    :param actions_selected: A dictionary mapping actions to the number of times this action was selected by the solver
    :param sum_reward: The accumulated reward after all plays
    """
    bandit_optimal_action = bandit.get_optimal_action()
    solver_optimal_action = solver.get_greedy_action()

    print('action bandit solver optimal selections')
    for action in range(NUM_ACTIONS):
        optimal_string = '{} {}'.format(
            'B' if action == bandit_optimal_action else ' ',
            'S' if action == solver_optimal_action else ' '
        )
        print('{action:^6d} {bandit:^ 6.3f} {solver:^ 6.3f} {optimal:^7} {selections:^ 10d}'.format(
            action=action,
            bandit=bandit.get_mean_rewards()[action],
            solver=solver.get_action_values()[action],
            optimal=optimal_string,
            selections=actions_selected[action]
        ))
    print('reward sum: {}'.format(sum_reward))


if __name__ == '__main__':
    main()
