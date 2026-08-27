import math
import matplotlib.pyplot as plt


def multi_feature_GD(self, heros, np, k=1, learning_rate=0.01):

    """
    Update all quantitative rule bounds from one shared gradient pass
    
    :param self: Rule whose quantitative conditions are being optimized
    :param heros: HEROS object containing training data and feature ranges
    :param np: numpy
    :param k: Sigmoid steepness parameter.
    :param learning_rate:  gradient-descent step size
    :return: None
    """

    def sigmoid(x):
        if x >= 0:
            z = math.exp(-x)
            return 1 / (1 + z)
        else:
            z = math.exp(x)
            return z / (1 + z)

    def sigmoid_derivative(x):
        s = sigmoid(x)
        return s * (1 - s)

    train_data = heros.env.train_data
    instance_states = train_data[0]
    outcomes = train_data[1]

    # Instances currently matching the rule
    valid_indices = [
        i for i, instance in enumerate(instance_states)
        if self.rule_matches_instance(instance, np)
    ]

    if len(valid_indices) == 0:
        return

    filtered_states = [instance_states[i] for i in valid_indices]
    filtered_outcomes = [outcomes[i] for i in valid_indices]

    n = len(filtered_states)

    # Compute p_i for every instance
    interval_probs = []
    feature_sigmoids = []

    for instance in filtered_states:

        p = 1.0
        sigs = []

        for pos, feature in enumerate(self.condition_indexes):

            lower = self.condition_values[pos][0]
            upper = self.condition_values[pos][1]

            x = instance[feature]

            s_lower = sigmoid(k * (x - lower))
            s_upper = sigmoid(k * (upper - x))

            sigs.append((s_lower, s_upper))

            p *= s_lower * s_upper

        feature_sigmoids.append(sigs)
        interval_probs.append(p)

    N = sum(interval_probs)

    if N == 0:
        return

    
    # Convert weighted class counts into normalized soft class probabilities.
    unique_classes = set(filtered_outcomes)

    class_probs = {}

    for c in unique_classes:
        class_probs[c] = (
            sum(interval_probs[i]
                for i in range(n)
                if filtered_outcomes[i] == c)
            / N
        )

    
    # Compute gradients for every feature
    new_bounds = []

    for pos, feature in enumerate(self.condition_indexes):

        lower = self.condition_values[pos][0]
        upper = self.condition_values[pos][1]

        dpi_lower = []
        dpi_upper = []

        for i, instance in enumerate(filtered_states):

            x = instance[feature]
            p = interval_probs[i]

            s_lower, s_upper = feature_sigmoids[i][pos]

            # Avoid divide-by-zero
            if s_lower < 1e-12:
                s_lower = 1e-12

            if s_upper < 1e-12:
                s_upper = 1e-12

            dp_lower = (
                p
                * (-k * sigmoid_derivative(k * (x - lower)))
                / s_lower
            )

            dp_upper = (
                p
                * (k * sigmoid_derivative(k * (upper - x)))
                / s_upper
            )

            dpi_lower.append(dp_lower)
            dpi_upper.append(dp_upper)

        sum_lower = sum(dpi_lower)
        sum_upper = sum(dpi_upper)

        dqc_lower = {}
        dqc_upper = {}

        for c in unique_classes:

            soft_lower = sum(
                dpi_lower[i]
                for i in range(n)
                if filtered_outcomes[i] == c
            )

            soft_upper = sum(
                dpi_upper[i]
                for i in range(n)
                if filtered_outcomes[i] == c
            )

            dqc_lower[c] = (
                soft_lower / N
                - class_probs[c] * sum_lower / N
            )

            dqc_upper[c] = (
                soft_upper / N
                - class_probs[c] * sum_upper / N
            )

        d_gini_lower = -2 * sum(
            class_probs[c] * dqc_lower[c]
            for c in unique_classes
        )

        d_gini_upper = -2 * sum(
            class_probs[c] * dqc_upper[c]
            for c in unique_classes
        )

        effective_lr = learning_rate * (heros.env.feat_q_range[feature][1]- heros.env.feat_q_range[feature][0])

        new_lower = lower - effective_lr * d_gini_lower
        new_upper = upper - effective_lr * d_gini_upper

        
        # Repair Ranges
        for instance in filtered_states:

            x = instance[feature]

            if x < new_lower:
                new_lower = x

            if x > new_upper:
                new_upper = x
        
        if new_lower > new_upper:
            new_lower, new_upper = new_upper, new_lower

        new_bounds.append((new_lower, new_upper))

    # Apply threshold updates
    for pos, (lower, upper) in enumerate(new_bounds):
        self.condition_values[pos][0] = lower
        self.condition_values[pos][1] = upper



