import math
import copy
import matplotlib.pyplot as plt


def update_soft_gini(self, instance_state, outcome_state, feature_index, k=1, decay=0.01):
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))


    x = instance_state[feature_index]
    position = self.condition_indexes.index(feature_index)
    lower, upper = self.condition_values[position]
    p = sigmoid(k*(x - lower)) * sigmoid(k*(upper - x))
    
    if feature_index not in self.soft_N:
        self.soft_N[feature_index] = 0.0
        self.soft_class_sum[feature_index] = {}
    

    self.soft_N[feature_index] += p
    self.soft_F[feature_index] += p * (outcome_state == self.action)


def Soft_Gini_SGD(self, heros, feature_index, np, random, k=1, learning_rate=0.01):
    """
    Calculate derivatives of gini with respect to lower and upper bounds.
    
    :param feature_index: The feature index to optimize
    :param k: Sigmoid steepness parameter
    :param learning_rate: The learning rate for gradient descent
    :return: Dictionary with updated lower and upper bounds
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

    #add output
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

    if (feature_index not in self.condition_indexes )or (feature_index in heros.cat_feature_indexes):
        return {'new_lower': None, 'new_upper': None, 'current_gini': 1.0}

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

    valid_indices = [
        i for i, instance in enumerate(instance_states)
        if self.rule_matches_instance(heros, instance, np)
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
        p = sigmoid(k * (feature_value - rule_lower)) * sigmoid(k * (rule_upper - feature_value))
        interval_probs.append(p)

    N = sum(interval_probs)
    if N == 0:
        return {'new_lower': rule_lower, 'new_upper': rule_upper, 'current_gini': 1.0}
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
        term1_lower = -k * sigmoid_derivative(k * (feature_value - rule_lower))
        term2_lower = sigmoid(k * (rule_upper - feature_value))
        dpi_d_lower.append(term1_lower * term2_lower)

        term1_upper = sigmoid(k * (feature_value - rule_lower))
        term2_upper = k * sigmoid_derivative(k * (rule_upper - feature_value))
        dpi_d_upper.append(term1_upper * term2_upper)

    # Step 3: Calculate dqc/d(lower) and dqc/d(upper)
    # dqc/d(bound) = (1/N) * Σ(dpi/d(bound) * 1[yi = c]) - (qc/N) * Σ(dpi/d(bound))

    sum_dpi_lower = sum(dpi_d_lower)
    sum_dpi_upper = sum(dpi_d_upper)

    dqc_d_lower = {}
    dqc_d_upper = {}

    for class_c in unique_classes:
        soft_correct_deriv_lower = sum(
            dpi_d_lower[i] for i in range(len(filtered_outcomes))
            if filtered_outcomes[i] == class_c
        )
        dqc_d_lower[class_c] = (soft_correct_deriv_lower / N) - (class_probs[class_c] / N) * sum_dpi_lower

        soft_correct_deriv_upper = sum(
            dpi_d_upper[i] for i in range(len(filtered_outcomes))
            if filtered_outcomes[i] == class_c
        )
        dqc_d_upper[class_c] = (soft_correct_deriv_upper / N) - (class_probs[class_c] / N) * sum_dpi_upper
    
    # Step 4: Calculate dGINI/d(bound)
    # GINI = 1 - Σ qc²
    # dGINI/d(bound) = -Σ (2 * qc * dqc/d(bound))
    
    d_gini_d_lower = -2 * sum(
        class_probs[class_c] * dqc_d_lower[class_c]
        for class_c in unique_classes
    )

    d_gini_d_upper = -2 * sum(
        class_probs[class_c] * dqc_d_upper[class_c]
        for class_c in unique_classes
    )

    feat_min, feat_max = heros.env.feat_q_range[feature_index]
    span = max(abs(feat_max - feat_min), 1.0)
    step_scale = learning_rate * span

    new_lower = rule_lower - step_scale * d_gini_d_lower
    new_upper = rule_upper - step_scale * d_gini_d_upper

    new_lower, new_upper = project_bounds(new_lower, new_upper)

    #add output
    #add comments
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
                width = max(1e-6, 0.05 * max(abs(feat_max - feat_min), 1.0))
                new_lower = max(feat_min, anchor_value - width)
                new_upper = min(feat_max, anchor_value + width)

    self.condition_values[position][0] = new_lower
    self.condition_values[position][1] = new_upper
    self.condition_values[position].sort()

    current_gini = 1.0 - sum(qc ** 2 for qc in class_probs.values())
    return {'new_lower': self.condition_values[position][0], 'new_upper': self.condition_values[position][1], 'current_gini': current_gini}


def optimize_quantitative_range(self, instance_state, quant_feat_list,heros,random,np):
    """Mutate the value range of a specified quantitative feature in a rule."""
    changed = False
    while not changed and len(quant_feat_list) > 0:
        feat = random.sample(quant_feat_list, 1)[0]
        quant_feat_list.remove(feat)
        if instance_state[feat] is not None:
            rule_position = self.condition_indexes.index(feat)
            new_bounds = Soft_Gini_SGD(self, heros, feat, np, random)

            if new_bounds is not None:
                self.condition_values[rule_position][0] = new_bounds["new_lower"]
                self.condition_values[rule_position][1] = new_bounds["new_upper"]
                self.condition_values[rule_position].sort()
                changed = True
    return feat


def delta_rule_mutation(self,instance_state, outcome_state, quant_feat_list, heros, random, np, lr=0.01):
    """

    """
    changed = False
    while not changed and len(quant_feat_list) > 0:
        feat = random.sample(quant_feat_list,1)[0]
        quant_feat_list.remove(feat)
        if instance_state[feat] != None:
            rule_position = self.condition_indexes.index(feat)

            lower =  self.condition_values[rule_position][0]
            upper = self.condition_values[rule_position][1]

            x = instance_state[feat]
            correct = (outcome_state == self.action)

            dist_lower = x - lower
            dist_upper = upper - x
            
            if correct:
                # contract slightly toward this instance 
                if dist_lower < dist_upper:
                    lower += lr * dist_lower * 0.1  # nudge lower up slightly, staying below x
                else:
                    upper -= lr * dist_upper * 0.1
            else:
                # this instance is incorrectly matched — push the nearer boundary away from it
                if dist_lower < dist_upper:
                    lower -= lr * (1 - dist_lower)  # push lower boundary up to exclude x
                else:
                    upper += lr * (1 - dist_upper)          
            


            self.condition_values[rule_position] = [lower, upper]   

            self.condition_values[rule_position].sort()

            '''
            #Ensure value range matches current instance's feature value
            if not self.condition_values[rule_position][0] < instance_state[feat] < self.condition_values[rule_position][1]:
                #Repair range to include current instance's feature value
                if self.condition_values[rule_position][1] - instance_state[feat] > instance_state[feat] - self.condition_values[rule_position][0]: #instance value closer to low end
                    self.condition_values[rule_position][0] = instance_state[feat]
                else:
                    self.condition_values[rule_position][1] = instance_state[feat]
            # Check for changing boundaries to infinity
            if self.condition_values[rule_position][0] < heros.env.feat_q_range[feat][0]: # if value range goes below that observed in training data, set low to negative infinity
                self.condition_values[rule_position][0] = -np.inf
            if self.condition_values[rule_position][1] > heros.env.feat_q_range[feat][1]: # if value range goes above that observed in training data, set high to positive infinity
                self.condition_values[rule_position][1] = np.inf
            '''
            changed = True       

    return feat


def tree_split(self, heros, feature_index, np, random, k=1, learning_rate=0.01):
    """
    Find the best decision-tree-style Gini split for feature_index.


    Parameters
    ----------
    self :LCS rule/classifier object.

    heros :environment containing train_data and feature ranges.

    feature_index :Quantitative feature to split.

    np :NumPy module, retained from original interface.

    random : Retained from original interface.

    k : Retained from original interface; not used.

    learning_rate :
        Retained from original interface; not used.

    Returns
    -------
    dict
        {
            'new_lower': updated lower bound,
            'new_upper': updated upper bound,
            'current_gini': Gini impurity of retained rule instances
        }
    """

    if feature_index in heros.cat_feature_indexes:
        return

    # Gini impurity
    def gini_impurity(labels):
        if len(labels) == 0:
            return 0.0

        _, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)

        return 1.0 - np.sum(probabilities ** 2)

    # Feature must already have a quantitative condition in the rule.
    if feature_index not in self.condition_indexes:
        return {
            'new_lower': None,
            'new_upper': None,
            'current_gini': 1.0
        }

    position = self.condition_indexes.index(feature_index)

    # Current bounds are retained as fallback.
    rule_lower = float(self.condition_values[position][0])
    rule_upper = float(self.condition_values[position][1])

    # Use the SAME training-data inputs as the original function.
    train_data = heros.env.train_data
    instance_states = train_data[0]
    outcomes = train_data[1]

    # Find instances that match the existing rule, BUT exclude feature_index from the matching calculation.
    #
    # This is important because feature_index is the feature whose threshold we are currently trying to discover.
    #
    # All other rule conditions remain fixed/immutable.
    valid_indices = []

    for i, instance in enumerate(instance_states):

        matches = True

        '''
        what i need to do:

        for each instance:
            for each feature speciifed in the rule:
                is the current feature the one we're optimizing?
                if yes:
                    skip to the next feature
                if no:
                    is the feature catergorical or quantitative:
                    if catergotrical:
                        if the rule's value and the instances value are not equal:
                            matches = false
                    else:
                        if the instances value is not in the rule's bounds
                            matches = false
        '''

        for i in range(len(self.condition_indexes)):
            feat = self.condition_indexes[i]
            
            # Do not apply the current condition for the feature that we are trying to split.
            if feat == feature_index:
                continue

            value = instance[feat]

            if feat in heros.cat_feature_indexes:
                rule_value = self.condition_values[i]

                if value != rule_value:
                    matches = False

            else:
                lower = self.condition_values[i][0]
                upper = self.condition_values[i][1]

                # Missing value does not match the rule.
                if value is None:
                    matches = False
                    break

                try:
                    if np.isnan(value):
                        matches = False
                        break
                except (TypeError, ValueError):
                    pass

                # Existing conditions remain fixed.
                if not (lower <= value <= upper):
                    matches = False
                    break

        if matches:
            valid_indices.append(i)

    # No instances matched the parent rule.
    if len(valid_indices) == 0:
        return {
            'new_lower': rule_lower,
            'new_upper': rule_upper,
            'current_gini': 1.0
        }

    # Pull out feature values and outcomes for the matched training instances.
    #
    # These are the observations on which the decision-tree-style split is evaluated.
    feature_values = []
    filtered_outcomes = []

    for i in valid_indices:

        value = instance_states[i][feature_index]

        if value is None:
            continue

        try:
            if np.isnan(value):
                continue
        except (TypeError, ValueError):
            pass

        feature_values.append(float(value))
        filtered_outcomes.append(outcomes[i])

    feature_values = np.asarray(feature_values, dtype=float)
    filtered_outcomes = np.asarray(filtered_outcomes)

    # Need at least two distinct feature values to split.
    if len(feature_values) == 0:
        return {
            'new_lower': rule_lower,
            'new_upper': rule_upper,
            'current_gini': 1.0
        }

    unique_values = np.unique(feature_values)

    if len(unique_values) < 2:
        current_gini = gini_impurity(filtered_outcomes)

        return {
            'new_lower': rule_lower,
            'new_upper': rule_upper,
            'current_gini': current_gini
        }

    # Sort feature values and corresponding outcomes together.
    order = np.argsort(feature_values)

    sorted_values = feature_values[order]
    sorted_outcomes = filtered_outcomes[order]

    # Generate decision-tree-style candidate thresholds.
    #
    # Thresholds occur between consecutive DISTINCT observed values.
    candidate_thresholds = []

    for i in range(len(sorted_values) - 1):

        left_value = sorted_values[i]
        right_value = sorted_values[i + 1]

        if left_value == right_value:
            continue

        threshold = (left_value + right_value) / 2.0

        candidate_thresholds.append(threshold)

    # Search for the threshold with the BEST weighted Gini.
    best_threshold = None
    best_weighted_gini = np.inf

    best_left_gini = None
    best_right_gini = None

    best_left_count = 0
    best_right_count = 0

    for threshold in candidate_thresholds:

        # Left child:
        #     feature <= threshold
        #
        # Right child:
        #     feature > threshold
        left_mask = sorted_values <= threshold
        right_mask = sorted_values > threshold

        left_count = np.sum(left_mask)
        right_count = np.sum(right_mask)

        if left_count == 0 or right_count == 0:
            continue

        left_outcomes = sorted_outcomes[left_mask]
        right_outcomes = sorted_outcomes[right_mask]

        left_gini = gini_impurity(left_outcomes)
        right_gini = gini_impurity(right_outcomes)

        total_count = left_count + right_count

        # Decision-tree weighted child impurity
        weighted_gini = (
            (left_count / total_count) * left_gini
            + (right_count / total_count) * right_gini
        )

        # retain the threshold with minimum weighted Gini.
        if weighted_gini < best_weighted_gini:

            best_weighted_gini = weighted_gini
            best_threshold = threshold

            best_left_gini = left_gini
            best_right_gini = right_gini

            best_left_count = left_count
            best_right_count = right_count

    # No valid split found.
    if best_threshold is None:
        current_gini = gini_impurity(filtered_outcomes)

        return {
            'new_lower': rule_lower,
            'new_upper': rule_upper,
            'current_gini': current_gini
        }

    # ------------------------------------------------------------
    # OPTION B:
    #
    # The threshold was chosen using BOTH children.
    #
    # Now retain only ONE side as the specialized LCS rule.
    #
    # If left and right have different Gini values, keep the purer side.
    #
    # If they are exactly tied, keep the side with more instances.
    # ------------------------------------------------------------
    if best_left_gini < best_right_gini:

        new_lower = -np.inf
        new_upper = float(best_threshold)

        current_gini = float(best_left_gini)

    elif best_right_gini < best_left_gini:

        new_lower = float(best_threshold)
        new_upper = np.inf

        current_gini = float(best_right_gini)

    else:

        # Equal Gini -> retain greater coverage.
        if best_left_count >= best_right_count:

            new_lower = -np.inf
            new_upper = float(best_threshold)

            current_gini = float(best_left_gini)

        else:

            new_lower = float(best_threshold)
            new_upper = np.inf

            current_gini = float(best_right_gini)

    # Update the rule.
    #
    # This intentionally stores +/- infinity instead of projecting
    # the condition into heros.env.feat_q_range.
    self.condition_values[position][0] = new_lower
    self.condition_values[position][1] = new_upper

    # Ensure [low, high] ordering.
    self.condition_values[position].sort()

    # Return exactly the same output fields as the original method.
    return {
        'new_lower': self.condition_values[position][0],
        'new_upper': self.condition_values[position][1],
        'current_gini': current_gini
    }