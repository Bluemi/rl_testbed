import numpy as np
import matplotlib.pyplot as plt

from n_armed_bandit import NArmedBandit


def main():
    avg_rewards = np.load('experiment2.dat')

    bandit = NArmedBandit(10, 42)

    # solver_names = ['epsilon greedy 0.01', 'epsilon greedy 0.1', 'epsilon greedy 0.1 with decay 5e-5']
    solver_names = ['epsilon greedy 0.01', 'optimistic initial values', 'upper confidence bound c=1']

    for avg_reward, solver_name in zip(avg_rewards, solver_names):
        plt.plot(avg_reward, label=f'{solver_name}', alpha=0.7)
    plt.plot([bandit.get_optimal_value()] * 5000, label='optimal value', color='black', linestyle='--')
    plt.axis((-10, 5000, 0.0, 1.8))
    plt.ylabel('reward')
    plt.legend(loc='best')
    plt.axhline(0, color='black')
    plt.show()


if __name__ == '__main__':
    main()
