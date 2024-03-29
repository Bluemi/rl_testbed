from functools import partial
from multiprocessing.pool import ThreadPool, Pool
from typing import Callable, List

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from base_solver import Solver, GreedySolver
from solvers.epsilon_greedy_solver import EpsilonGreedySolver
from upper_bound_confidence_solver import UpperBoundConfidenceSolver
from n_armed_bandit import NArmedBandit
from training import train, TrainResult
from update_rules import sample_average_update_rule, create_weighted_average_update_rule

NUM_ACTIONS = 10
NUM_PLAYS_PER_TRAINING = 5000
NUM_TRAININGS = 2000


def solver_to_results(
        solver_constructor: Callable[[], Solver],
        bandit: NArmedBandit,
        num_trainings
) -> List[TrainResult]:
    """
    Takes a solver constructor and trains it NUM_TRAININGS times on the given bandit.

    :param solver_constructor: A callable that creates a solver
    :param bandit: The bandit to train on
    :param num_trainings: Specifies how often the solver is trained on the bandit
    :return: The results of the training
    """
    results = []
    solver_name = str(solver_constructor())
    for _ in tqdm(range(num_trainings), desc=f'training solver {solver_name}: '):
        solver = solver_constructor()
        result = train(bandit, solver, NUM_PLAYS_PER_TRAINING)
        results.append(result)
    return results


def results_to_rewards(results: List[List[TrainResult]]) -> np.ndarray:
    """
    Converts a list of lists of train results into a 3D numpy array with the following shape.
    rewards[solver_index][training_index][sample_index] = reward

    :param results: The training results for multiple solvers
    :return: An numpy array containing the rewards of the given train results
    """
    rewards = np.zeros((len(results), NUM_TRAININGS, NUM_PLAYS_PER_TRAINING), dtype=np.float)
    for solver_index, solver_results in enumerate(results):
        for training_index, solver_result in tqdm(
            enumerate(solver_results),
            desc=f'transforming rewards for solver {solver_results[0].solver}',
            total=NUM_TRAININGS
        ):
            for sample_index, reward in enumerate(solver_result.rewards):
                rewards[solver_index][training_index][sample_index] = reward
    return rewards


def main():
    bandit = NArmedBandit(NUM_ACTIONS, seed=42)

    solver_constructors = [
        lambda: EpsilonGreedySolver(
            name='epsilon greedy 0.01',
            n=NUM_ACTIONS,
            update_rule=sample_average_update_rule,
            epsilon=0.01
        ),
        # lambda: EpsilonGreedySolver(
        #     name='epsilon greedy 0.1',
        #     n=NUM_ACTIONS,
        #     update_rule=sample_average_update_rule,
        #     epsilon=0.1
        # ),
        # lambda: EpsilonGreedySolver(
        #     name='epsilon greedy 0.1 with decay 5e-5',
        #     n=NUM_ACTIONS,
        #     update_rule=sample_average_update_rule,
        #     epsilon=0.1,
        #     epsilon_decay=5e-5
        # ),
        lambda: GreedySolver(
            name='optimistic initial values',
            n=NUM_ACTIONS,
            update_rule=create_weighted_average_update_rule(0.3),
            initial_action_value=10,
        ),
        lambda: UpperBoundConfidenceSolver(
            name='upper confidence bound c=1',
            n=NUM_ACTIONS,
            update_rule=sample_average_update_rule,
            confidence=1
        ),
    ]

    results: List[List[TrainResult]] = []
    for solver_constructor in solver_constructors:
        result = solver_to_results(solver_constructor, bandit=bandit, num_trainings=NUM_TRAININGS)
        results.append(result)

    rewards = results_to_rewards(results)
    avg_rewards = np.mean(rewards, axis=1)

    print(f'optimal value: {bandit.get_optimal_value()}')
    print(f'mean values: {np.mean(avg_rewards, axis=1)}')

    avg_rewards.dump('experiment2.dat')

    for avg_reward, solver_constructor in zip(avg_rewards, solver_constructors):
        solver = solver_constructor()
        plt.plot(avg_reward, label=f'{solver}', alpha=0.7)
    plt.plot([bandit.get_optimal_value()]*NUM_PLAYS_PER_TRAINING, label='optimal value', color='black', linestyle='--')
    plt.axis((0, 5010, 0.0, 1.8))
    plt.ylabel('reward')
    plt.legend(loc='best')
    plt.axhline(0, color='black')
    plt.show()


if __name__ == '__main__':
    main()
