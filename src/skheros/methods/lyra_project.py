from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
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

            modified_complete_rule_evaluation_class(
                left_rule, heros, random, None
            )

            modified_complete_rule_evaluation_class(
                right_rule, heros, random, None
            )

        elif heros.outcome_type == 'quant':

            modified_complete_rule_evaluation_quant(
                left_rule, heros
            )

            modified_complete_rule_evaluation_quant(
                right_rule, heros
            )

        # Select the better candidate
        if left_rule.fitness >= right_rule.fitness:
            optimized_conditions[selected_feature] = left_candidate
        else:
            optimized_conditions[selected_feature] = right_candidate

    # ---------------------------------------------------------
    # Apply ALL optimized conditions to the rule at the end
    # ---------------------------------------------------------

    for feature, optimized_condition in optimized_conditions.items():

        condition_position = rule.condition_indexes.index(feature)

        rule.condition_values[condition_position] = optimized_condition

    return rule
