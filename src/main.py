from n_armed_bandit import NArmedBandit
from solvers.epsilon_greedy_solver import EpsilonGreedySolver
from training import compare_once
from update_rules import sample_average_update_rule, create_weighted_average_update_rule

NUM_ACTIONS = 10
NUM_PLAYS_PER_TRAINING = 500


def main():
    bandit = NArmedBandit(NUM_ACTIONS)
    solvers = [
        EpsilonGreedySolver(
            name='sample average solver',
            n=NUM_ACTIONS,
            update_rule=sample_average_update_rule,
            epsilon=0.1
        ),
        EpsilonGreedySolver(
            name='weighted average solver',
            n=NUM_ACTIONS,
            update_rule=create_weighted_average_update_rule(0.2),
            epsilon=0.1
        ),
        EpsilonGreedySolver(
            name='optimistic solver',
            n=NUM_ACTIONS,
            update_rule=create_weighted_average_update_rule(0.15),
            epsilon=0.0,
            initial_action_value=3.0
        )
    ]

    solver_ranking = compare_once(bandit, solvers, NUM_PLAYS_PER_TRAINING)

    for solver, result in solver_ranking:
        print('{:<27}: {}'.format(str(solver), result.reward_sum))


if __name__ == '__main__':
    main()
