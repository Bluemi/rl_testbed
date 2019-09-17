from typing import Callable


def sample_average_update_rule(current_action_value: float, reward: float, number_of_action_tries: int) -> float:
    """
    Implements the sample average update rule with
    new_action_value = old_action_value + (1.0 / (number_of_action_tries + 1)) * (reward - current_action_value)

    :param current_action_value: The current approximation of the value for an action
    :param reward: The reward that was achieved by playing this action
    :param number_of_action_tries: The number this action was tried before this play
    :return: The new value of the action after applying the given reward.
    """
    return current_action_value + (1.0 / (number_of_action_tries + 1)) * (reward - current_action_value)


def create_weighted_average_update_rule(alpha: float) -> Callable[[float, float, int], float]:
    """
    Creates a new weighted_average_update_rule with the given alpha step size parameter

    :param alpha: The step size parameter of the created weighted average update rule
    :return: A weighted average update rule
    """
    def weighted_average_update_rule(current_action_value: float, reward: float, _number_of_action_tries: int):
        return _weighted_average_update_rule(current_action_value, reward, alpha)
    return weighted_average_update_rule


def _weighted_average_update_rule(current_action_value: float, reward: float, alpha: float) -> float:
    """
    Implements the weighted average update rule with
    new_action_value = old_action_value + alpha * (reward - current_action_value)

    :param current_action_value: The current approximation of the value for an action
    :param reward: The reward that was achieved by playing this action
    :param alpha: The step size parameter of this weighted average update rule
    :return: The new value of the action after applying the given reward.
    """
    return current_action_value + alpha * (reward - current_action_value)
