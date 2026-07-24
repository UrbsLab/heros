import math
import matplotlib.pyplot as plt


def multi_feature_GD(self, heros, np, k=1, learning_rate=0.01):
    #Performs gradient descent for all of a rule's features simultaneously

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

    m = len(self.condition_indexes)
    n = len(filtered_states)

    plot_history = list(getattr(self, "_gradient_arrow_history", []))
    max_samples = getattr(self, "_gradient_arrow_sample_limit", 10)
    plot_already_shown = getattr(self, "_gradient_arrow_plot_shown", False)

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

    
    # Class probabilities
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

    
    # Aggregate the gradient direction across features and keep a small sample
    if m > 0:
        gradient_dx = sum(bound[0] - self.condition_values[pos][0] for pos, bound in enumerate(new_bounds)) / m
        gradient_dy = sum(bound[1] - self.condition_values[pos][1] for pos, bound in enumerate(new_bounds)) / m

        plot_history.append((gradient_dx, gradient_dy))

        if len(plot_history) >= max_samples and not plot_already_shown:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.axhline(0, color="gray", lw=0.7, alpha=0.6)
            ax.axvline(0, color="gray", lw=0.7, alpha=0.6)

            for dx, dy in plot_history:
                ax.arrow(
                    0,
                    0,
                    0.05 * dx,
                    0.05 * dy,
                    head_width=0.01,
                    length_includes_head=True,
                    color="C0",
                    alpha=0.7,
                )

            ax.scatter(0, 0, color="black")
            ax.set_xlabel("dG/d(lower)")
            ax.set_ylabel("dG/d(upper)")
            ax.set_title("Sampled Soft Gini GD updates")
            ax.grid(alpha=0.3)
            plt.show()
            plt.close(fig)
            plot_already_shown = True
            plot_history = []

        self._gradient_arrow_history = plot_history
        self._gradient_arrow_plot_shown = plot_already_shown

    # Apply threshold updates
    for pos, (lower, upper) in enumerate(new_bounds):
        self.condition_values[pos][0] = lower
        self.condition_values[pos][1] = upper



