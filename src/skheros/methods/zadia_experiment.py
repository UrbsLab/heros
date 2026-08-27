import math
from src.skheros.methods.zadia_project import Soft_Gini_SGD

def compute_soft_gini(self, heros, np, k=1):
    """
    Calculate the rule's Gini impurity using soft interval membership

    Each matching instance receives a weight formed by multiplying the
    sigmoid membership of every quantitative condition in the rule

    :param self: Rule whose conditions and matching logic are evaluated.
    :param heros: HEROS object containing the training data
    :param np: numpy
    :param k: Sigmoid steepness parameter
    :return: Soft Gini impurity as a float. 1 when no usable weight exists
    """

    def sigmoid(x):
        if x >= 0:
            z = math.exp(-x)
            return 1 / (1 + z)
        else:
            z = math.exp(x)
            return z / (1 + z)

    train_data = heros.env.train_data
    instance_states = train_data[0]
    outcomes = train_data[1]

    # Keep the objective aligned with the optimizer's matching population.
    valid_indices = [
        i for i, instance in enumerate(instance_states)
        if self.rule_matches_instance(instance, np)
    ]

    if len(valid_indices) == 0:
        return 1.0

    filtered_states = [instance_states[i] for i in valid_indices]
    filtered_outcomes = [outcomes[i] for i in valid_indices]

    # Multiply condition memberships to get each instance's whole-rule weight.
    interval_probs = []

    for instance in filtered_states:

        p = 1.0

        for position, feature in enumerate(self.condition_indexes):

            lower = self.condition_values[position][0]
            upper = self.condition_values[position][1]

            x = instance[feature]

            p *= (
                sigmoid(k * (x - lower))
                * sigmoid(k * (upper - x))
            )

        interval_probs.append(p)

    N = sum(interval_probs)

    if N == 0:
        return 1.0

    # Normalize weighted class counts and calculate 1 - sum(class_prob^2).
    unique_classes = set(filtered_outcomes)

    gini = 1.0

    for c in unique_classes:

        soft_count = sum(
            interval_probs[i]
            for i in range(len(filtered_outcomes))
            if filtered_outcomes[i] == c
        )

        qc = soft_count / N

        gini -= qc * qc

    return gini


def multi_iter_GD(self, heros, random, np, max_epochs = 20):
    """
    Run repeated per-feature soft-Gini updates until convergence

    Each epoch updates every condition and checks both objective change and the largest boundary movement against stopping tolerances

    :param self: Rule whose quantitative bounds are optimized in place
    :param heros: HEROS object containing training data and feature ranges
    :param random: Random module
    :param np: numpy
    :param max_epochs: Maximum number of optimization epochs
    :return: None
    """

    prev_gini = compute_soft_gini(self, heros, np)

    for i in range(max_epochs):

        max_change = 0
 
        # Update one interval at a time using the current rule state.
        for feature in self.condition_indexes:

            position = self.condition_indexes.index(feature)

            old_lower = self.condition_values[position][0]
            old_upper = self.condition_values[position][1]

            Soft_Gini_SGD(self, heros,feature, np)

            new_lower = self.condition_values[position][0]
            new_upper = self.condition_values[position][1]

            max_change = max(max_change, abs(new_lower - old_lower), abs(new_upper - old_upper))

        new_gini = compute_soft_gini(self, heros, np)

        if abs(new_gini - prev_gini) < 1e-4:
            break

        if max_change < 1e-3:
            break

        prev_gini = new_gini


