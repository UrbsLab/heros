from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from scipy.stats import chi2
from functools import lru_cache
import copy

def local_quantitative_optmization(rule, target_state, heros, random, np):
    """optimize the conditions for quantitative features in a rule.
    For rules born by covering or evolution, target_state is the current training instance and the side of
    each split containing its value is kept. For rules initialized by the decision tree, target_state is None
    (there is no current instance) and the side giving the higher rule fitness is kept."""

    # identify quantitative features in the rule
    quant_features = []

    for feat in rule.condition_indexes:
        if heros.env.feat_types[feat] == 0:
            quant_features.append(feat)

    # return the rule if there are no features to optimize
    if len(quant_features) == 0:
        return rule

    # loop through each quantitative feature in the rule
    while len(quant_features) > 0:

        # randomly select a feature to optimize
        selected_feature = random.choice(quant_features)
        quant_features.remove(selected_feature)


        # stores training data indices that match the rule without the feature to be optimized
        optimization_indices = []

        # loop through instances in the training data
        for instance_index in range(heros.env.num_instances):

            instance_state = heros.env.train_data[0][instance_index]

            # check if instance matches the rule (without the rule to be optimized)
            if matches_except_feature(rule, instance_state, heros, selected_feature):
                optimization_indices.append(instance_index)

        # skip if no training instances match the rest of the rule (nothing to fit a threshold on)
        if len(optimization_indices) == 0:
            continue

        # build the optimization dataset
        X_opt = [] # values for the feature we are optimizing
        y_opt = [] # outcomes

        for instance_index in optimization_indices:

            # pull the training data instance matching according to the previously identified indices
            instance_state = heros.env.train_data[0][instance_index] 

            # pull the relevant feature and outcome values
            feature_value = instance_state[selected_feature]
            outcome_value = heros.env.train_data[1][instance_index]

            X_opt.append(feature_value)
            y_opt.append(outcome_value)

        # reformat for DecisionTreeClassifier
        X_opt = np.array(X_opt).reshape(-1, 1)
        y_opt = np.array(y_opt)

        if heros.outcome_type == 'class':
            tree = DecisionTreeClassifier(max_depth = 1, random_state = heros.random_state)
        elif heros.outcome_type == 'quant':
            tree = DecisionTreeRegressor(max_depth = 1, random_state = heros.random_state)
        
        tree.fit(X_opt, y_opt)

        threshold = tree.tree_.threshold[0]

        # skip if no threshold was found
        if threshold == -2:
            continue

        # candidate conditions for each side of the split (sklearn sends x <= threshold to the left)
        left_condition = [-np.inf, threshold]
        right_condition = [threshold, np.inf]

        # index of the condition to update
        condition_position = rule.condition_indexes.index(selected_feature)

        if target_state is not None:
            # rule born by covering or evolution: keep the side of the split that contains the current instance's value
            target_value = target_state[selected_feature]
            if target_value is None:  # missing value: no side to choose, leave the condition as it is
                continue

            if target_value <= threshold:
                new_condition = left_condition
            else:
                new_condition = right_condition
        else:
            # rule initialized by the decision tree: keep the side of the split with the higher rule fitness
            new_condition = select_condition_by_fitness(rule, condition_position, left_condition, right_condition, heros, random)

        # update the condition for the selected feature
        rule.condition_values[condition_position] = new_condition

    return rule


def local_quantitative_optimization_v4(rule, target_state, heros, random, np, max_passes=5):
    """repeat local_quantitative_optmization passes over the rule's quantitative features, so each feature's split is
    refit on the instances matching the other features' updated conditions. Stops once a full pass leaves every
    condition unchanged, or after max_passes passes."""

    for _ in range(max_passes):

        # copy the conditions so in-place updates during the pass can be detected
        previous_conditions = [list(value) if isinstance(value, list) else value for value in rule.condition_values]

        rule = local_quantitative_optmization(rule, target_state, heros, random, np)

        # stop once the conditions have converged
        if rule.condition_values == previous_conditions:
            break

    return rule


def select_condition_by_fitness(rule, condition_position, left_condition, right_condition, heros, random):
    """evaluate a copy of the rule with each candidate condition and return the condition giving the higher fitness.
    Copies are evaluated so the original rule's statistics are left untouched."""

    candidates = []
    for condition in (left_condition, right_condition):
        candidate = copy.deepcopy(rule)
        candidate.condition_values[condition_position] = condition

        # evaluation adds to these counts, so start each candidate from zero
        candidate.match_cover = 0
        candidate.correct_cover = 0

        if heros.outcome_type == 'class':
            modified_complete_rule_evaluation_class(candidate, heros, random, None)
        elif heros.outcome_type == 'quant':
            modified_complete_rule_evaluation_quant(candidate, heros)
        candidates.append(candidate)

    # compare fitness (ties go to the left side)
    if candidates[0].fitness >= candidates[1].fitness:
        return left_condition
    return right_condition



def modified_complete_rule_evaluation_class(self,heros,random,target_instance_outcome):
        """ Evaluates rule performance across the entire training dataset and update rule parameters accordingly. **NO PARETO FRONT UPDATING** """
        # See if rule matches and if it's correct for each instance in the training data (updating match and correct counts)
        front_updated = False #only actively used by pareto front rule-fitness
        train_data = heros.env.train_data
        self.instance_outcome_count = {}
        for each in heros.env.classes:
            self.instance_outcome_count[each] = 0
        for instance_index in range(heros.env.num_instances):
            instance_state = train_data[0][instance_index] 
            outcome_state = train_data[1][instance_index]
            if self.match(instance_state,heros):
                self.match_cover += 1
                self.instance_outcome_count[outcome_state] += 1 #update rule's outcome counts for training set
        # Assign class that yields highest rule accuracy - regardless of current target instance class
        self.instance_outcome_prop = copy.deepcopy(self.instance_outcome_count)
        #Convert class counts first into class accuracies then into 'useful' accuracies to take class imbalance into account
        for each in self.instance_outcome_prop:
            self.instance_outcome_prop[each] /= self.match_cover
        class_balanced_accuracies = copy.deepcopy(self.instance_outcome_prop)
        for each in class_balanced_accuracies:
            class_balanced_accuracies[each] = (class_balanced_accuracies[each] - (1-heros.env.class_weights[each]))/(heros.env.class_weights[each])
        #Choose action (i.e. 'class') that maximizes matching rule's accuracy
        high_val = max(class_balanced_accuracies.values())
        candidate_actions = [k for k, v in class_balanced_accuracies.items() if v == high_val]
        if target_instance_outcome in candidate_actions: #go with the current target instance class if there is a tie for best class
            self.action = target_instance_outcome
        else:
            self.action = random.choice(sorted(candidate_actions))
        self.correct_cover = self.instance_outcome_count[self.action]
        # Calculate rule accuracy ***************************
        try:
            self.accuracy = self.correct_cover / float(self.match_cover)
        except:
            self.accuracy = 0.0
        # Calculate useful accuracy *************************
        class_probability = 1 - heros.env.class_weights[self.action]
        self.useful_accuracy = (self.accuracy - class_probability) / (1 - class_probability)
        if self.useful_accuracy < 0.0:
            self.useful_accuracy = 0.0
        # Calculate useful coverage *************************
        self.useful_coverage = self.correct_cover - (self.match_cover * class_probability)
        if self.useful_coverage < 0.0:
            self.useful_coverage = 0.0
        # Calculate fitness *********************************
        if heros.fitness_function == 'accuracy':
            self.fitness = pow(self.accuracy, heros.nu)
        elif heros.fitness_function == 'pareto':
            
            # Calculate and update the rule fitness
            if len(heros.rule_pareto.rule_front) == 0: #Pareto front not yet built (e.g. optimizing during the first covering event)
                self.fitness = None
            else:
                self.fitness = heros.rule_pareto.get_pareto_fitness(self.useful_accuracy,self.useful_coverage, False,heros)
            if self.fitness is None: #Pareto front only has (0,0) for useful_accuracy and useful_coverage
                self.fitness = pow(self.accuracy, heros.nu)
        else:
            print("Fitness metric not available.")
        
        #self.encoding = self.encode_rule_binary(heros.env.num_feat)
        return False

def modified_complete_rule_evaluation_quant(self,heros):
        """ Evaluates rule performance across the entire training dataset and update rule parameters accordingly. **NO PARETO FRONT UPDATING"""
        # See if rule matches and if it's correct for each instance in the training data (updating match and correct counts)
        front_updated = False #only actively used by pareto front rule-fitness
        train_data = heros.env.train_data
        heros.env.num_instances
        #Find matching and correct instances and sum the outcome states within matching rules 
        instance_match_list = []
        self.prediction = 0
        for instance_index in range(heros.env.num_instances):
            instance_state = train_data[0][instance_index] 
            outcome_state = train_data[1][instance_index]
            if self.match(instance_state, heros):
                instance_match_list.append(instance_index)
                self.match_cover += 1
                self.prediction += outcome_state #sum of matching instance outcomes becomes the rule's quantitative prediction (like with decision trees)
                if self.action[0] <= outcome_state <= self.action[1]:
                    self.correct_cover += 1
        # Calculate rule accuracy ***************************
        try:
            self.accuracy = self.correct_cover / float(self.match_cover)
        except:
            self.accuracy = 0.0
        # Convert sum of instance outcomes to an average (will serve as rule's )
        self.prediction = self.prediction / float(self.match_cover)
        # Calculate rule's mean absolute error on training data
        self.mean_absolute_error = 0
        for instance_index in instance_match_list:
            instance_state = train_data[0][instance_index] 
            outcome_state = train_data[1][instance_index]
            self.mean_absolute_error += abs(outcome_state - self.prediction)
        self.mean_absolute_error = self.mean_absolute_error / float(self.match_cover)
        # Calculate useful accuracy (for quanatiative outcomes) *************************
        self.set_outcome_range_probability(heros)   
        if self.outcome_range_prob == 1: #outcome range captures all training instance outcomes (i.e. useless outcome range)
            self.useful_accuracy = 0.0
            self.useful_coverage = 0.0
        else:
            self.useful_accuracy = (self.accuracy - self.outcome_range_prob) / (1 - self.outcome_range_prob)
            if self.useful_accuracy < 0.0:
                self.useful_accuracy = 0.0
            # Calculate useful coverage *************************
            self.useful_coverage = self.correct_cover - (self.match_cover * self.outcome_range_prob)
            if self.useful_coverage < 0.0:
                self.useful_coverage = 0.0
        # Calculate fitness *********************************
        if heros.fitness_function == 'accuracy' or heros.nu > 1:
            self.fitness = pow(self.useful_accuracy, heros.nu)
        elif heros.fitness_function == 'pareto':
            
            # Calculate and update the rule fitness
            if len(heros.rule_pareto.rule_front) == 0: #Pareto front not yet built (e.g. optimizing during the first covering event)
                self.fitness = None
            else:
                self.fitness = heros.rule_pareto.get_pareto_fitness(self.useful_accuracy,self.useful_coverage, False,heros)
            if self.fitness is None: #Pareto front only has (0,0) for useful_accuracy and useful_coverage
                self.fitness = pow(self.useful_accuracy, heros.nu)
        else:
            print("Fitness metric not available.")
        return False



def matches_except_feature(self, instance_state, heros, excluded_feature):
    """determine if a data instance matches all rule conditions except one specified feature"""

    # loop through conditions in the rule
    for i in range(len(self.condition_indexes)):
        feat = self.condition_indexes[i]

        # skip if looking at the feature we are optimizing 
        if feat == excluded_feature:
            continue

        is_categorical = heros.env.feat_types[feat]
        value = instance_state[feat]

        # match check if feature is categorical 
        if is_categorical:

            if value is None:
                return False
            
            elif value == self.condition_values[i]:
                pass

            else:
                return False

        # match check if feature is quantitative
        else:

            if value is None: 
                return False

            elif self.condition_values[i][0] <= value <=self.condition_values[i][1]:
                pass

            else:
                return False

    return True




def local_quantitative_optimization_v2(rule, x, Y, heros, random, np):
    """optimize the conditions for quantitative features in a rule, apply all rule updates at the end"""

    # identify quantitative features in the rule
    quant_features = []

    for feat in rule.condition_indexes:
        if heros.env.feat_types[feat] == 0:
            quant_features.append(feat)

    # Return the rule if there are no quantitative features to optimize
    if len(quant_features) == 0:
        return rule

    # Store optimized conditions here.
    # The original rule is not modified during optimization.
    optimized_conditions = {}

    # Loop through each quantitative feature
    while len(quant_features) > 0:

        # Randomly select a feature to optimize
        selected_feature = random.choice(quant_features)
        quant_features.remove(selected_feature)

        # Store training data indices that match the rule
        # without the feature being optimized
        optimization_indices = []

        for instance_index in range(heros.env.num_instances):

            instance_state = heros.env.train_data[0][instance_index]

            if matches_except_feature(rule, 
                instance_state,
                heros,
                selected_feature
            ):
                optimization_indices.append(instance_index)

        # Build optimization dataset
        X_opt = []
        y_opt = []

        for instance_index in optimization_indices:

            instance_state = heros.env.train_data[0][instance_index]

            feature_value = instance_state[selected_feature]
            outcome_value = heros.env.train_data[1][instance_index]

            X_opt.append(feature_value)
            y_opt.append(outcome_value)

        # Reformat for DecisionTreeClassifier
        X_opt = np.array(X_opt).reshape(-1, 1)
        y_opt = np.array(y_opt)

        # Build shallow decision tree
        tree = DecisionTreeClassifier(
            max_depth=1,
            random_state=heros.random_state
        )

        tree.fit(X_opt, y_opt)

        threshold = tree.tree_.threshold[0]

        # Skip if no threshold was found
        if threshold == -2:
            continue

        # Build candidate conditions
        left_candidate = [-np.inf, threshold]
        right_candidate = [threshold, np.inf]

        # Index of the condition being optimized
        condition_position = rule.condition_indexes.index(selected_feature)

        # Create candidate rules from the ORIGINAL rule
        left_rule = copy.deepcopy(rule)
        right_rule = copy.deepcopy(rule)

        # Apply candidate condition
        left_rule.condition_values[condition_position] = left_candidate
        right_rule.condition_values[condition_position] = right_candidate

        # Evaluate candidate rules
        if heros.outcome_type == 'class':

            modified_complete_rule_evaluation_class(left_rule, heros, random, None)

            modified_complete_rule_evaluation_class(right_rule, heros, random, None)

        elif heros.outcome_type == 'quant':

            modified_complete_rule_evaluation_quant(left_rule, heros)

            modified_complete_rule_evaluation_quant(right_rule, heros)

        # Select the better candidate
        if left_rule.fitness >= right_rule.fitness:
            optimized_conditions[selected_feature] = left_candidate
        else:
            optimized_conditions[selected_feature] = right_candidate

    # Apply ALL optimized conditions to the rule at the end

    for feature, optimized_condition in optimized_conditions.items():
        condition_position = rule.condition_indexes.index(feature)
        rule.condition_values[condition_position] = optimized_condition

    return rule


def gini_split_threshold(feature_values, outcomes, np):
    """find the best depth-1 Gini split threshold for one feature, matching DecisionTreeClassifier(max_depth = 1)
    without sklearn's per-fit overhead. Returns None when there is no split (pure node or constant feature)."""

    # sklearn fits trees on float32 values, so cast the same way to get identical thresholds
    values = np.asarray(feature_values, dtype=np.float32).astype(np.float64)
    outcomes = np.asarray(outcomes)

    # missing values (None -> nan) are left out of the split
    keep = ~np.isnan(values)
    values = values[keep]
    outcomes = outcomes[keep]

    # sort feature values and outcomes together
    order = np.argsort(values, kind='stable')
    values = values[order]
    classes, class_codes = np.unique(outcomes[order], return_inverse=True)

    # no split if the node is pure or the feature has only one value
    if len(classes) < 2 or values[0] == values[-1]:
        return None

    # class counts on the left of each split position (left = first p instances) and on the right
    left_counts = np.cumsum(np.eye(len(classes))[class_codes], axis=0)[:-1]
    right_counts = left_counts[-1] + np.eye(len(classes))[class_codes[-1]] - left_counts
    left_n = np.arange(1, len(values))
    right_n = len(values) - left_n

    # maximizing this proxy is the same as minimizing the weighted Gini impurity of the two children (sklearn's proxy)
    proxy = (left_counts ** 2).sum(axis=1) / left_n + (right_counts ** 2).sum(axis=1) / right_n

    # only split between distinct values (sklearn's FEATURE_THRESHOLD), ties go to the first (lowest) position as in sklearn
    valid = values[1:] > values[:-1] + 1e-7
    proxy[~valid] = -np.inf
    p = int(np.argmax(proxy)) + 1

    # midpoint threshold, computed as sklearn does
    threshold = values[p - 1] / 2.0 + values[p] / 2.0
    if threshold == values[p] or np.isinf(threshold):
        threshold = values[p - 1]

    return threshold


def local_quantitative_optimization_v3(rule, target_state, heros, random, np):
    """same as local_quantitative_optmization, but finds each split threshold with a numpy Gini split
    (gini_split_threshold) instead of fitting a sklearn decision tree. Class outcomes only."""

    # identify quantitative features in the rule
    quant_features = []

    for feat in rule.condition_indexes:
        if heros.env.feat_types[feat] == 0:
            quant_features.append(feat)

    # return the rule if there are no features to optimize
    if len(quant_features) == 0:
        return rule

    # loop through each quantitative feature in the rule
    while len(quant_features) > 0:

        # randomly select a feature to optimize
        selected_feature = random.choice(quant_features)
        quant_features.remove(selected_feature)


        # stores training data indices that match the rule without the feature to be optimized
        optimization_indices = []

        # loop through instances in the training data
        for instance_index in range(heros.env.num_instances):

            instance_state = heros.env.train_data[0][instance_index]

            # check if instance matches the rule (without the rule to be optimized)
            if matches_except_feature(rule, instance_state, heros, selected_feature):
                optimization_indices.append(instance_index)

        # skip if no training instances match the rest of the rule (nothing to fit a threshold on)
        if len(optimization_indices) == 0:
            continue

        # build the optimization dataset
        X_opt = [] # values for the feature we are optimizing
        y_opt = [] # outcomes

        for instance_index in optimization_indices:

            # pull the training data instance matching according to the previously identified indices
            instance_state = heros.env.train_data[0][instance_index]

            # pull the relevant feature and outcome values
            feature_value = instance_state[selected_feature]
            outcome_value = heros.env.train_data[1][instance_index]

            X_opt.append(feature_value)
            y_opt.append(outcome_value)

        # find the best Gini split threshold
        threshold = gini_split_threshold(X_opt, y_opt, np)

        # skip if no threshold was found
        if threshold is None:
            continue

        # candidate conditions for each side of the split (x <= threshold goes to the left, as in sklearn)
        left_condition = [-np.inf, threshold]
        right_condition = [threshold, np.inf]

        # index of the condition to update
        condition_position = rule.condition_indexes.index(selected_feature)

        if target_state is not None:
            # rule born by covering or evolution: keep the side of the split that contains the current instance's value
            target_value = target_state[selected_feature]
            if target_value is None:  # missing value: no side to choose, leave the condition as it is
                continue

            if target_value <= threshold:
                new_condition = left_condition
            else:
                new_condition = right_condition
        else:
            # rule initialized by the decision tree: keep the side of the split with the higher rule fitness
            new_condition = select_condition_by_fitness(rule, condition_position, left_condition, right_condition, heros, random)

        # update the condition for the selected feature
        rule.condition_values[condition_position] = new_condition

    return rule


def training_arrays(heros, np):
    """return the training features as a float matrix (missing values as nan) and the class outcomes as integer codes
    (indexes into heros.env.classes). Built once per training run and cached on heros.env."""

    if getattr(heros.env, 'lqo_arrays', None) is None:
        X = np.array([[np.nan if value is None else value for value in row] for row in heros.env.train_data[0]], dtype=np.float64)
        class_index = {outcome: i for i, outcome in enumerate(heros.env.classes)}
        class_codes = np.array([class_index[outcome] for outcome in heros.env.train_data[1]], dtype=np.int64)
        heros.env.lqo_arrays = (X, class_codes)

    return heros.env.lqo_arrays


def condition_mask(feat, condition, X, heros):
    """boolean mask of training instances matching one rule condition (missing values never match)"""

    column = X[:, feat]
    if heros.env.feat_types[feat]:  # categorical
        return column == condition
    return (column >= condition[0]) & (column <= condition[1])  # comparisons with nan are False


def rule_match_mask(rule, X, heros, np, excluded_feature=None):
    """boolean mask of training instances matching every rule condition except excluded_feature"""

    mask = np.ones(X.shape[0], dtype=bool)
    for feat, condition in zip(rule.condition_indexes, rule.condition_values):
        if feat != excluded_feature:
            mask &= condition_mask(feat, condition, X, heros)
    return mask


def match_mask_fitness(match_mask, class_codes, heros, np, target_outcome):
    """fitness a rule with this match mask would receive from complete_rule_evaluation_class: the action is the class with the
    highest class-balanced accuracy (ties go to target_outcome, otherwise the lowest sorted class instead of a random choice),
    and pareto fitness is taken against a copy of the front updated with the rule, so the real front is left untouched.
    Returns -inf if the rule matches no training instances."""

    counts = np.bincount(class_codes[match_mask], minlength=len(heros.env.classes))
    return class_counts_fitness(counts, heros, np, target_outcome)


def class_counts_fitness(counts, heros, np, target_outcome):
    """match_mask_fitness computed from the per-class counts of the matched instances (counts[i] is the number of matched
    instances of class heros.env.classes[i]). Returns -inf if the rule matches no training instances."""

    match_cover = int(counts.sum())
    if match_cover == 0:
        return -np.inf

    classes = heros.env.classes
    class_weights = heros.env.class_weights

    # choose the action the same way complete_rule_evaluation_class does
    balanced_accuracies = [(counts[i] / match_cover - (1 - class_weights[c])) / class_weights[c] for i, c in enumerate(classes)]
    high_val = max(balanced_accuracies)
    candidate_actions = [c for c, v in zip(classes, balanced_accuracies) if v == high_val]
    action = target_outcome if target_outcome in candidate_actions else sorted(candidate_actions)[0]

    correct_cover = counts[classes.index(action)]
    accuracy = correct_cover / float(match_cover)
    class_probability = 1 - class_weights[action]
    useful_accuracy = max(0.0, (accuracy - class_probability) / (1 - class_probability))
    useful_coverage = max(0.0, correct_cover - match_cover * class_probability)

    if heros.fitness_function == 'pareto':
        fitness = pareto_fitness_with_candidate(useful_accuracy, useful_coverage, heros)
        if fitness is None:  # front only has (0,0)
            fitness = pow(accuracy, heros.nu)
        return fitness
    return pow(accuracy, heros.nu)


def pareto_fitness_with_candidate(useful_accuracy, useful_coverage, heros):
    """pareto fitness of a rule with these metrics, taken against the rule front as it would be after update_front with the
    rule, without changing the real front. When the rule is already on the front or is dominated by a front point,
    update_front leaves the front unchanged, so the real front is used directly; only when the rule would join the front is
    a copy updated (the copy is the slow part)."""

    front = heros.rule_pareto
    candidate = (useful_accuracy, useful_coverage)
    objectives = ['max', 'max']
    if front.rule_front and (candidate in front.rule_front or any(front.dominates(point, candidate, objectives) for point in front.rule_front)):
        return front.get_pareto_fitness(useful_accuracy, useful_coverage, False, heros)

    front = copy.deepcopy(front)
    front.update_front(useful_accuracy, useful_coverage, objectives)
    return front.get_pareto_fitness(useful_accuracy, useful_coverage, False, heros)


def local_quantitative_optimization_v5(rule, target_state, heros, random, np, target_outcome=None):
    """like local_quantitative_optimization_v3, but a condition only changes when the change strictly improves rule fitness.
    For each quantitative feature (in random order) the candidates are: the current condition, the side(s) of the Gini split,
    and removing the condition (only while the rule keeps at least one other condition). For rules born by covering or
    evolution (target_state given) only the split side containing the current instance's value is a candidate, so the rule
    still matches the current instance. For rules initialized by the decision tree (target_state None) both sides are.
    Candidates are scored with match_mask_fitness on the full training set. Ties between improving candidates go to removal,
    then the left side. Class outcomes only."""

    if heros.outcome_type != 'class':
        raise ValueError("local_quantitative_optimization_v5 supports class outcomes only")

    # identify quantitative features in the rule
    quant_features = [feat for feat in rule.condition_indexes if heros.env.feat_types[feat] == 0]

    # return the rule if there are no features to optimize
    if len(quant_features) == 0:
        return rule

    X, class_codes = training_arrays(heros, np)

    # loop through each quantitative feature in the rule
    while len(quant_features) > 0:

        # randomly select a feature to optimize
        selected_feature = random.choice(quant_features)
        quant_features.remove(selected_feature)
        condition_position = rule.condition_indexes.index(selected_feature)

        # instances matching the rule without the feature to be optimized
        others_mask = rule_match_mask(rule, X, heros, np, excluded_feature=selected_feature)

        # skip if no training instances match the rest of the rule (nothing to fit a threshold on)
        if not others_mask.any():
            continue

        # fitness of the rule as it is
        current_condition = rule.condition_values[condition_position]
        current_fitness = match_mask_fitness(others_mask & condition_mask(selected_feature, current_condition, X, heros), class_codes, heros, np, target_outcome)

        # candidate conditions (None means remove the condition)
        candidates = []
        if len(rule.condition_indexes) > 1:
            candidates.append(None)

        threshold = gini_split_threshold(X[others_mask, selected_feature], class_codes[others_mask], np)
        if threshold is not None:
            left_condition = [-np.inf, threshold]
            right_condition = [threshold, np.inf]
            if target_state is None:
                candidates.extend([left_condition, right_condition])
            elif target_state[selected_feature] is not None:
                # keep the side of the split that contains the current instance's value
                candidates.append(left_condition if target_state[selected_feature] <= threshold else right_condition)

        # find the best candidate
        best_condition = None
        best_fitness = -np.inf
        for condition in candidates:
            if condition is None:
                candidate_mask = others_mask
            else:
                candidate_mask = others_mask & condition_mask(selected_feature, condition, X, heros)
            fitness = match_mask_fitness(candidate_mask, class_codes, heros, np, target_outcome)
            if fitness > best_fitness:
                best_condition = condition
                best_fitness = fitness

        # only change the rule if fitness strictly improves
        if best_fitness <= current_fitness:
            continue

        if best_condition is None:
            rule.condition_indexes.pop(condition_position)
            rule.condition_values.pop(condition_position)
        else:
            rule.condition_values[condition_position] = best_condition

    return rule


def local_quantitative_optimization_v6(rule, target_state, heros, random, np, target_outcome=None, vacuous_tolerance=0.02):
    """like local_quantitative_optmization (sklearn depth-1 tree split), followed by two checks on each optimized condition:
    (1) if the optimized condition gives lower rule fitness than the original condition, the original condition is kept;
    (2) otherwise, if the optimized condition still matches at least (1 - vacuous_tolerance) of the instances matching the rest
    of the rule, it is vacuous and is removed (unless it is the rule's only condition, in which case the optimized condition
    is kept). Fitness is scored with match_mask_fitness on the full
    training set. For rules born by covering or evolution (target_state given) the split side containing the current
    instance's value is kept; for rules initialized by the decision tree (target_state None) the side with the higher fitness
    is kept. Class outcomes only."""

    if heros.outcome_type != 'class':
        raise ValueError("local_quantitative_optimization_v6 supports class outcomes only")

    # identify quantitative features in the rule
    quant_features = [feat for feat in rule.condition_indexes if heros.env.feat_types[feat] == 0]

    # return the rule if there are no features to optimize
    if len(quant_features) == 0:
        return rule

    X, class_codes = training_arrays(heros, np)

    # loop through each quantitative feature in the rule
    while len(quant_features) > 0:

        # randomly select a feature to optimize
        selected_feature = random.choice(quant_features)
        quant_features.remove(selected_feature)
        condition_position = rule.condition_indexes.index(selected_feature)

        # instances matching the rule without the feature to be optimized
        others_mask = rule_match_mask(rule, X, heros, np, excluded_feature=selected_feature)

        # fit the split on those instances, leaving out missing values for the selected feature
        fit_mask = others_mask & ~np.isnan(X[:, selected_feature])

        # skip if no training instances match the rest of the rule (nothing to fit a threshold on)
        if not fit_mask.any():
            continue

        tree = DecisionTreeClassifier(max_depth = 1, random_state = heros.random_state)
        tree.fit(X[fit_mask, selected_feature].reshape(-1, 1), class_codes[fit_mask])
        threshold = tree.tree_.threshold[0]

        # skip if no threshold was found
        if threshold == -2:
            continue

        # candidate conditions for each side of the split (sklearn sends x <= threshold to the left)
        left_condition = [-np.inf, threshold]
        right_condition = [threshold, np.inf]
        left_mask = others_mask & condition_mask(selected_feature, left_condition, X, heros)
        right_mask = others_mask & condition_mask(selected_feature, right_condition, X, heros)

        if target_state is not None:
            # rule born by covering or evolution: keep the side of the split that contains the current instance's value
            target_value = target_state[selected_feature]
            if target_value is None:  # missing value: no side to choose, leave the condition as it is
                continue

            if target_value <= threshold:
                new_condition, new_mask = left_condition, left_mask
            else:
                new_condition, new_mask = right_condition, right_mask
            new_fitness = match_mask_fitness(new_mask, class_codes, heros, np, target_outcome)
        else:
            # rule initialized by the decision tree: keep the side of the split with the higher rule fitness (ties go to the left side)
            left_fitness = match_mask_fitness(left_mask, class_codes, heros, np, target_outcome)
            right_fitness = match_mask_fitness(right_mask, class_codes, heros, np, target_outcome)
            if left_fitness >= right_fitness:
                new_condition, new_mask, new_fitness = left_condition, left_mask, left_fitness
            else:
                new_condition, new_mask, new_fitness = right_condition, right_mask, right_fitness

        # check 1: does the optimized condition degrade rule fitness compared to the original condition? if so, keep the original
        original_mask = others_mask & condition_mask(selected_feature, rule.condition_values[condition_position], X, heros)
        if new_fitness < match_mask_fitness(original_mask, class_codes, heros, np, target_outcome):
            continue

        # check 2: is the optimized condition vacuous (excludes too few of the instances matching the rest of the rule)?
        if len(rule.condition_indexes) > 1 and new_mask.sum() >= (1 - vacuous_tolerance) * others_mask.sum():
            rule.condition_indexes.pop(condition_position)
            rule.condition_values.pop(condition_position)
            continue

        # update the condition for the selected feature
        rule.condition_values[condition_position] = new_condition

    return rule


def resolve_num_quantiles(num_quantiles, num_instances):
    """bin count from num_quantiles: an integer is used as is, a float in (0, 1) is a fraction of the number of training
    instances (rounded, at least 2 bins)"""

    if isinstance(num_quantiles, float):
        return max(2, int(round(num_quantiles * num_instances)))
    return num_quantiles


def quantile_cut_points(heros, np, num_quantiles):
    """return {feature index: sorted array of cut points} for every quantitative feature, where the cut points split the
    feature's training values (missing values left out) into num_quantiles quantile bins, e.g. num_quantiles = 10 gives the
    10th, 20th, ..., 90th percentiles. num_quantiles may also be a float in (0, 1), a fraction of the number of training
    instances (see resolve_num_quantiles). Duplicate cut points (from tied values) are merged. Built once per training run
    and cached on heros.env."""

    cache = getattr(heros.env, 'lqo_cut_points', None)
    if cache is None or cache[0] != num_quantiles:
        X, _ = training_arrays(heros, np)
        num_bins = resolve_num_quantiles(num_quantiles, X.shape[0])
        heros.env.lqo_num_bins = num_bins
        probabilities = np.arange(1, num_bins) / num_bins
        cut_points = {}
        for feat in range(X.shape[1]):
            if heros.env.feat_types[feat] == 0:  # quantitative
                column = X[:, feat]
                column = column[~np.isnan(column)]
                cut_points[feat] = np.unique(np.quantile(column, probabilities)) if column.size > 0 else np.array([])
        heros.env.lqo_cut_points = (num_quantiles, cut_points)

    return heros.env.lqo_cut_points[1]


@lru_cache(maxsize=None)
def chi_square_critical_value(alpha, degrees_of_freedom):
    """chi-square statistic needed for significance at level alpha (upper tail)"""

    return chi2.isf(alpha, degrees_of_freedom)


def local_quantitative_optimization_v7(rule, target_state, heros, random, np, target_outcome=None, max_passes=5):
    """repeat v7_single_pass over the rule's quantitative conditions until a full pass leaves the rule unchanged (or for
    max_passes passes), so each condition is judged against the final state of the others. Without this, a condition
    optimized early in a pass can become redundant once a later condition is tightened, and would never be re-tested."""

    for _ in range(max_passes):

        # copy the conditions so in-place updates during the pass can be detected
        previous_indexes = list(rule.condition_indexes)
        previous_values = [list(value) if isinstance(value, list) else value for value in rule.condition_values]

        rule = v7_single_pass(rule, target_state, heros, random, np, target_outcome)

        # stop once the conditions have converged
        if rule.condition_indexes == previous_indexes and rule.condition_values == previous_values:
            break

    return rule


def v7_single_pass(rule, target_state, heros, random, np, target_outcome=None):
    """one pass of local_quantitative_optimization_v7: optimize each quantitative condition by exhaustive search over quantile intervals. For each quantitative feature
    (in random order), the candidate bounds are -inf, the feature's quantile cut points (see quantile_cut_points, number of
    bins set by heros.num_quantiles) and +inf. Every [lower, upper] pair with lower < upper splits the training instances
    that match the rest of the rule (and have a value for the feature) into inside and outside the interval, and the
    interval with the lowest weighted Gini impurity of that split replaces the condition (rule fitness and the pareto front
    are not used). Ties go to the interval matching more instances (the more general one), then the lower interval.
    Only intervals containing at least heros.min_interval_instances of those instances are candidates. The best interval is
    kept only if its inside/outside split is significant: Pearson chi-square test of the inside/outside x class table at
    heros.significance_alpha, Bonferroni corrected for the number of candidate intervals. If it is not significant, or no
    interval is large enough, the condition is dropped, even if it is the rule's last condition (the caller then discards
    the rule, since a rule with no conditions matches every instance).
    For rules born by covering or evolution (target_state given) only intervals containing the current instance's value are
    candidates, so the rule still matches the current instance; a feature whose value is missing in the current instance is
    left as it is. target_outcome is unused (kept so v7 is called like v5 and v6). Class outcomes only."""

    if heros.outcome_type != 'class':
        raise ValueError("local_quantitative_optimization_v7 supports class outcomes only")

    # identify quantitative features in the rule
    quant_features = [feat for feat in rule.condition_indexes if heros.env.feat_types[feat] == 0]

    # return the rule if there are no features to optimize
    if len(quant_features) == 0:
        return rule

    X, class_codes = training_arrays(heros, np)
    cut_points = quantile_cut_points(heros, np, getattr(heros, 'num_quantiles', 10))
    num_classes = len(heros.env.classes)
    significance_alpha = getattr(heros, 'significance_alpha', 0.05)
    min_interval_instances = max(1, getattr(heros, 'min_interval_instances', 10))

    # loop through each quantitative feature in the rule
    while len(quant_features) > 0:

        # randomly select a feature to optimize
        selected_feature = random.choice(quant_features)
        quant_features.remove(selected_feature)
        condition_position = rule.condition_indexes.index(selected_feature)

        # rules born by covering or evolution must keep matching the current instance
        target_value = None
        if target_state is not None:
            target_value = target_state[selected_feature]
            if target_value is None:  # missing value: no interval to require, leave the condition as it is
                continue

        # feature values and classes of the instances matching the rest of the rule (missing values never match a condition)
        others_mask = rule_match_mask(rule, X, heros, np, excluded_feature=selected_feature)
        values = X[others_mask, selected_feature]
        codes = class_codes[others_mask]
        present = ~np.isnan(values)
        values = values[present]
        codes = codes[present]

        # skip if no training instances match the rest of the rule
        if values.size == 0:
            continue

        # candidate interval bounds
        bounds = np.concatenate(([-np.inf], cut_points[selected_feature], [np.inf]))

        # cumulative per-class counts over the sorted values: cumulative[k] = class counts of the k smallest values
        order = np.argsort(values, kind='stable')
        values = values[order]
        one_hot = np.eye(num_classes, dtype=np.int64)[codes[order]]
        cumulative = np.vstack((np.zeros((1, num_classes), dtype=np.int64), np.cumsum(one_hot, axis=0)))

        # class counts of values <= each bound and < each bound, so the counts in [bounds[i], bounds[j]] are
        # at_or_below[j] - below[i] (both ends inclusive, as in condition matching)
        at_or_below = cumulative[np.searchsorted(values, bounds, side='right')]
        below = cumulative[np.searchsorted(values, bounds, side='left')]

        # every [lower, upper] interval (lower index i < upper index j, ordered by i then j)
        lower_index, upper_index = np.triu_indices(len(bounds), k=1)
        if target_value is not None:
            # keep only intervals containing the current instance's value
            contains_target = (bounds[lower_index] <= target_value) & (bounds[upper_index] >= target_value)
            lower_index = lower_index[contains_target]
            upper_index = upper_index[contains_target]

        # per-class counts inside and outside each interval
        inside = at_or_below[upper_index] - below[lower_index]
        outside = cumulative[-1] - inside
        inside_n = inside.sum(axis=1)
        outside_n = outside.sum(axis=1)

        # minimizing weighted Gini impurity is the same as maximizing sum(inside^2)/inside_n + sum(outside^2)/outside_n
        # (an empty side adds nothing); intervals matching fewer than min_interval_instances instances are not candidates
        valid = inside_n >= min_interval_instances
        with np.errstate(divide='ignore', invalid='ignore'):
            score = np.where(valid, (inside ** 2).sum(axis=1) / inside_n, -np.inf) \
                  + np.where(outside_n > 0, (outside ** 2).sum(axis=1) / outside_n, 0.0)

        # is the best interval's split significant? Pearson chi-square test of the inside/outside x class table, with a
        # Bonferroni correction for the number of candidate intervals tried
        significant = False
        if valid.any():
            # best score, ties to the interval matching more instances, then the first (lowest) interval
            tied = np.flatnonzero(score == score.max())
            best = tied[inside_n[tied] == inside_n[tied].max()][0]
            if significance_alpha >= 1:  # test disabled: always keep the best interval
                significant = True
            elif outside_n[best] > 0:  # an interval holding every instance does not split them
                observed = np.vstack((inside[best], outside[best])).astype(np.float64)
                observed = observed[:, cumulative[-1] > 0]  # classes present among the matched instances
                degrees_of_freedom = observed.shape[1] - 1  # 0 if the matched instances are all one class
                if degrees_of_freedom > 0:
                    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / values.size
                    chi_square = ((observed - expected) ** 2 / expected).sum()
                    significant = chi_square >= chi_square_critical_value(significance_alpha / int(valid.sum()), degrees_of_freedom)

        # drop the condition if no interval is large enough or the best split is not significant, even if it is the rule's
        # last condition: a rule left with no conditions is discarded by the caller (covering, GA, tree initialization)
        if not significant:
            rule.condition_indexes.pop(condition_position)
            rule.condition_values.pop(condition_position)
            continue

        # update the condition for the selected feature
        rule.condition_values[condition_position] = [float(bounds[lower_index[best]]), float(bounds[upper_index[best]])]

    return rule
