def sample_average_update_rule(current_action_value, reward, number_of_action_tries) -> float:
    """
    Implements the sample average update rule with
    new_action_value = old_action_value + (1.0 / (number_of_action_tries + 1)) * (reward - current_action_value)

    :param current_action_value: The current approximation of the value for an action
    :param reward: The reward that was achieved by playing this action
    :param number_of_action_tries: The number this action was tried before this play
    :return: The new value of the action after applying the given reward.
    """
    return current_action_value + (1.0 / (number_of_action_tries + 1)) * (reward - current_action_value)
