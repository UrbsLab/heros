import math

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



def lower_sgd(self, heros, feature_index, np, random, k=1, learning_rate=0.01):
    #CODE FROM HERE IS FOR [LOWER, NP.INF] THRESHOLD

    if feature_index not in self.condition_indexes:
        return {'new_lower': None, 'new_upper': None, 'current_gini': 1.0}

    position = self.condition_indexes.index(feature_index)
    rule_lower = float(self.condition_values[position][0])

    train_data = heros.env.train_data
    instance_states = train_data[0]
    outcomes = train_data[1]


    valid_indices = [
        i for i, instance in enumerate(instance_states)
        if self.rule_matches_instance(instance, np)
    ]

    if len(valid_indices) == 0:
        print("No valid instances for this feature, returning current bounds and gini.")
        return {'new_lower': rule_lower, 'current_gini': 1.0}

    filtered_states = [instance_states[i] for i in valid_indices]
    filtered_outcomes = [outcomes[i] for i in valid_indices]


    # Step 1: Calculate pi, N, and qc (from previous method)
    interval_probs = []
    for instance in filtered_states:
        feature_value = instance[feature_index]
        p = sigmoid(k * (feature_value - rule_lower))
        interval_probs.append(p)

    N = sum(interval_probs)
    if N == 0:
        return {'new_lower': rule_lower,  'current_gini': 1.0}
        #May have something to do with setting the decent limit later on
    
    unique_classes = set(filtered_outcomes)
    class_probs = {}

    for class_c in unique_classes:
        soft_correct = sum(
            interval_probs[i] for i in range(len(filtered_outcomes))
            if filtered_outcomes[i] == class_c
        )
        class_probs[class_c] = soft_correct / N

    # Step 2: Calculate dpi/d(lower) for each instance

    dpi_d_lower = []


    for instance in filtered_states:
        feature_value = instance[feature_index]
        
        # dpi/d(lower) = -k * σ'(k*(xi - lower)) * σ(k*(upper - xi))
        term1_lower = -k * sigmoid_derivative(k * (feature_value - rule_lower))
        term2_lower = sigmoid(k * (rule_upper - feature_value))
        dpi_d_lower.append(term1_lower * term2_lower)

    # Step 3: Calculate dqc/d(lower) and dqc/d(upper)
    # dqc/d(bound) = (1/N) * Σ(dpi/d(bound) * 1[yi = c]) - (qc/N) * Σ(dpi/d(bound))

    sum_dpi_lower = sum(dpi_d_lower)

    dqc_d_lower = {}

    for class_c in unique_classes:
        soft_correct_deriv_lower = sum(
            dpi_d_lower[i] for i in range(len(filtered_outcomes))
            if filtered_outcomes[i] == class_c
        )
        dqc_d_lower[class_c] = (soft_correct_deriv_lower / N) - (class_probs[class_c] / N) * sum_dpi_lower
    
    # Step 4: Calculate dGINI/d(bound)
    # GINI = 1 - Σ qc²
    # dGINI/d(bound) = -Σ (2 * qc * dqc/d(bound))
    
    d_gini_d_lower = -2 * sum(
        class_probs[class_c] * dqc_d_lower[class_c]
        for class_c in unique_classes
    )


    feat_min, feat_max = heros.env.feat_q_range[feature_index]
    span = max(abs(feat_max - feat_min), 1.0)
    step_scale = learning_rate * span

    new_lower = rule_lower - step_scale * d_gini_d_lower

    return new_lower


def upper_sgd(self, heros, feature_index, np, random, k=1, learning_rate=0.01):
    #CODE FROM HERE IS FOR [-NP.INF, UPPER] THRESHOLD

    if feature_index not in self.condition_indexes:
        return {'new_lower': None, 'new_upper': None, 'current_gini': 1.0}

    position = self.condition_indexes.index(feature_index)
    rule_upper = float(self.condition_values[position][1])

    train_data = heros.env.train_data
    instance_states = train_data[0]
    outcomes = train_data[1]


    valid_indices = [
        i for i, instance in enumerate(instance_states)
        if self.rule_matches_instance(instance, np)
    ]

    if len(valid_indices) == 0:
        print("No valid instances for this feature, returning current bounds and gini.")
        return {'new_lower': rule_lower, 'new_upper': rule_upper, 'current_gini': 1.0}

    filtered_states = [instance_states[i] for i in valid_indices]
    filtered_outcomes = [outcomes[i] for i in valid_indices]


    
    # Step 1: Calculate pi, N, and qc (from previous method)
    interval_probs = []
    for instance in filtered_states:
        feature_value = instance[feature_index]
        p =  sigmoid(k * (rule_upper - feature_value))
        interval_probs.append(p)

    N = sum(interval_probs)
    if N == 0:
        return { 'new_upper': rule_upper, 'current_gini': 1.0}
        #May have something to do with setting the decent limit later on
    
    unique_classes = set(filtered_outcomes)
    class_probs = {}

    for class_c in unique_classes:
        soft_correct = sum(
            interval_probs[i] for i in range(len(filtered_outcomes))
            if filtered_outcomes[i] == class_c
        )
        class_probs[class_c] = soft_correct / N

    # Step 2: Calculate dpi/d(lower) and dpi/d(upper) for each instance

    dpi_d_lower = []
    dpi_d_upper = []

    for instance in filtered_states:
        feature_value = instance[feature_index]
        
        # dpi/d(lower) = -k * σ'(k*(xi - lower)) * σ(k*(upper - xi))
        term1_upper = sigmoid(k * (feature_value - rule_lower))
        term2_upper = k * sigmoid_derivative(k * (rule_upper - feature_value))
        dpi_d_upper.append(term1_upper * term2_upper)

    # Step 3: Calculate dqc/d(lower) and dqc/d(upper)
    # dqc/d(bound) = (1/N) * Σ(dpi/d(bound) * 1[yi = c]) - (qc/N) * Σ(dpi/d(bound))

    sum_dpi_upper = sum(dpi_d_upper)

    dqc_d_upper = {}

    for class_c in unique_classes:
        soft_correct_deriv_upper = sum(
            dpi_d_upper[i] for i in range(len(filtered_outcomes))
            if filtered_outcomes[i] == class_c
        )
        dqc_d_upper[class_c] = (soft_correct_deriv_upper / N) - (class_probs[class_c] / N) * sum_dpi_upper
    
    # Step 4: Calculate dGINI/d(bound)
    # GINI = 1 - Σ qc²
    # dGINI/d(bound) = -Σ (2 * qc * dqc/d(bound))

    d_gini_d_upper = -2 * sum(
        class_probs[class_c] * dqc_d_upper[class_c]
        for class_c in unique_classes
    )

    feat_min, feat_max = heros.env.feat_q_range[feature_index]
    span = max(abs(feat_max - feat_min), 1.0)
    step_scale = learning_rate * span

    new_upper = rule_upper - step_scale * d_gini_d_upper

    return new_upper


def one_side_sgd(self, heros, feature_index, np, random, k=1, learning_rate=0.01):
    """
    Calculate derivatives of gini with respect to either the lower or upper bound
    
    :param feature_index: The feature index to optimize
    :param k: Sigmoid steepness parameter
    :param learning_rate: The learning rate for gradient descent
    :return: Dictionary with updated lower and upper bounds
    """

    def project_bounds(lower, upper):
        feat_min, feat_max = heros.env.feat_q_range[feature_index]
        lower = float(lower)
        upper = float(upper)

        if feat_min is not None:
            lower = max(lower, feat_min)
        if feat_max is not None:
            upper = min(upper, feat_max)

        if upper <= lower:
            midpoint = 0.5 * (lower + upper)
            #may change
            width = max(abs(upper - lower), 1e-6)
            half_width = max(width, 0.01 * max(abs(feat_max - feat_min), 1.0))
            lower = midpoint - half_width
            upper = midpoint + half_width

            if feat_min is not None:
                lower = max(lower, feat_min)
            if feat_max is not None:
                upper = min(upper, feat_max)

        if upper <= lower:
            lower = feat_min if feat_min is not None else lower
            upper = feat_max if feat_max is not None else upper

        return lower, upper

    if feature_index not in self.condition_indexes:
        return {'new_lower': None, 'new_upper': None, 'current_gini': 1.0}

    feat_min, feat_max = heros.env.feat_q_range[feature_index]

    position = self.condition_indexes.index(feature_index)
    rule_lower = float(self.condition_values[position][0])
    rule_upper = float(self.condition_values[position][1])

    train_data = heros.env.train_data
    instance_states = train_data[0]
    outcomes = train_data[1]

    def interval_matches_any_instance(lower, upper):
        for instance in instance_states:
            value = instance[feature_index]

            if lower <= value <= upper:
                return True
        return False

    if random.random() < 0.5:
        new_lower = lower_sgd(self, heros, feature_index, np, random, k=1, learning_rate=0.01)
        new_upper = np.inf 
    else:
        new_lower = -np.inf
        new_upper = upper_sgd(self, heros, feature_index, np, random, k=1, learning_rate=0.01)

    new_lower, new_upper = project_bounds(new_lower, new_upper)

    if not interval_matches_any_instance(new_lower, new_upper):
        new_lower, new_upper = rule_lower, rule_upper
        if not interval_matches_any_instance(new_lower, new_upper):
            feat_values = [
                instance[feature_index]
                for instance in instance_states
                if instance[feature_index] is not None and not (isinstance(instance[feature_index], (float, int)) and np.isnan(instance[feature_index]))
            ]
            if feat_values:
                anchor_value = min(feat_values, key=lambda value: abs(value - 0.5 * (rule_lower + rule_upper)))
                width = max(1e-6, 0.01 * max(abs(feat_max - feat_min), 1.0))
                new_lower = max(feat_min, anchor_value - width)
                new_upper = min(feat_max, anchor_value + width)

    self.condition_values[position][0] = new_lower
    self.condition_values[position][1] = new_upper
    self.condition_values[position].sort()

    return {'new_lower': self.condition_values[position][0], 'new_upper': self.condition_values[position][1]}