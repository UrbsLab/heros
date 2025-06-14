import copy
import numpy as np

class RULE:
    def __init__(self):
        """ Initializes objects that define an individual rule. """
        # REFERENCE OBJECTS ******************************************************
        self.ID = None #unique identifier for this rule (used in model evolution and does not guarantee global uniqueness)
        # FIXED RULE PARAMETERS ***************************************************
        # Rule condition (IF-part of rule) 
        self.condition_indexes = [] #list of feature indexes from the dataset that are 'specified' in this rule
        self.condition_values = [] #list of feature values or value-ranges corresponding to the feature indexes in self.feature_index_list
        # Rule action (THEN-part of rule) 
        self.action = None #action value for this rule (i.e. action/outcome predicted by this rule)
        # Other fixed rule parameters *************************************************************
        self.accuracy = None #rule-accuracy (not to be confused with model accuracy) i.e. of the instances this rule matches, the proportion where this rule predicts the correct outcome
        self.match_cover = 0 #number of training instances matched by this rule
        self.correct_cover = 0 #number of training instances that both matched and had the same action/outcome as this rule
        self.birth_iteration = None #iteration number when this rule was first introduced (or re-introduced) to the population
        self.match_set = []
        # Non-standary metric parameters
        self.useful_accuracy = None
        self.useful_coverage = None
        self.outcome_range_prob = None
        self.mean_absolute_error = None
        self.prediction = None #average of training instance outcome values that match this rule (rule's specific prediction - similar to decision tree)
        # FLEXIBLE RULE PARAMETERS ***************************************************
        self.fitness = None #rule 'goodness' metric that drives many aspects of algorithm learning, discovery, and prediction
        self.numerosity = 1 #number of virtual copies of this rule maintained in the population - can protect rule from random deletion - increases influence of rule
        self.ave_match_set_size = 1 #average size of the match sets in which this rule was included across all training instances - used in deletion to promote niching
        self.deletion_prob = None #probability of rule being selected for deletion

    def __eq__(self, other):
        return isinstance(other, RULE) and self.ID == other.ID

    def __hash__(self):
        return hash(self.ID)
    
    def assign_ID(self, ID):
        self.ID = ID
    
    def initialize_by_covering(self,set_numerosity_sum,instance_state,outcome_state,heros,random):
        """ Initializes a rule using the covering mechanism. """
        self.birth_iteration = copy.deepcopy(heros.iteration)
        self.ave_match_set_size = set_numerosity_sum
        num_to_specify = random.randint(1,heros.rsl) #number of features to specify in rule introduced by covering - must be 1 to the number of features defined by the 'rule specificity limit (rsl)'
        self.specify_condition_covering(num_to_specify,instance_state,heros,random) #covering,num_to_specify,instance_state,changed_features
        if heros.outcome_type == 'class':
            self.action = outcome_state
        elif heros.outcome_type == 'quant':
            outcome_range = heros.env.outcome_range[1] - heros.env.outcome_range[0]
            range_radius = (outcome_range / 2.0)* random.randint(25,75)*0.01
            low = outcome_state - range_radius
            if low < heros.env.outcome_range[0]: # if value range goes below that observed in training data, set low to negative infinity
                low = -np.inf
            high = outcome_state + range_radius
            if high >heros.env.outcome_range[1]: # if value range goes above that observed in training data, set high to positive infinity
                high = np.inf
            self.action = [low,high] #ALKR Representation, Initialization centered around training instance with a range between 25 and 75% of the domain size.  
            self.quantitative_range_check_fix(self.action,outcome_state,random)  
        else:
            pass

    def initialize_by_parent(self,parent_rule,heros):
        """ Initializes a rule by copying a parent rule. """
        self.condition_indexes = copy.deepcopy(parent_rule.condition_indexes)
        self.condition_values = copy.deepcopy(parent_rule.condition_values) 
        self.action = copy.deepcopy(parent_rule.action)
        self.accuracy = None 
        self.useful_accuracy = None
        self.useful_coverage = 0
        self.match_cover = 0 
        self.correct_cover = 0 
        self.birth_iteration = copy.deepcopy(heros.iteration)
        self.fitness = None 
        self.numerosity = 1 
        self.ave_match_set_size = copy.deepcopy(parent_rule.ave_match_set_size) 
        self.deletion_prob = None 

    def match(self, instance_state, heros):
        """ Determine if rule matches the feature values of a given data instance. """
        for i in range(len(self.condition_indexes)): #for each feature specified in rule
            feat = self.condition_indexes[i]
            is_categorical = heros.env.feat_types[feat] # Assigned a value of 0 (False, Quantiative Feature) or 1 (True, Categorical Feature)
            value = instance_state[feat] #value of feature for current instance in dataset
            # Categorical feature ******************************************************
            if is_categorical: 
                if value is None: # Missing value in the dataset (by default, missing values do not match specified rule conditions)
                    return False
                elif value == self.condition_values[i]: #instance value matches condition value
                    pass
                else: #instance value does not match condition value
                    return False
            # Quanatiative feature ******************************************************
            else: 
                if value is None: # Missing value in the dataset (by default, missing values do not match specified rule conditions)
                    return False
                elif self.condition_values[i][0] <= value <= self.condition_values[i][1]: #instance value falls within condition value range
                    pass
                else: #instance value does not match condition value
                    return False  
        return True #rule matches instance
    
    def uniform_crossover(self,other_rule,heros,random):
        """ Apply uniform crossover between two new offspring rules """
        # Identify feature indexes that can be crossed over
        o1_condition_indexes = copy.deepcopy(self.condition_indexes)
        o2_condition_indexes = copy.deepcopy(other_rule.condition_indexes)
        crossover_canidate_indexes = copy.deepcopy(o1_condition_indexes)
        for feat in o2_condition_indexes:
            is_categorical = heros.env.feat_types[feat]
            if feat not in crossover_canidate_indexes:
                crossover_canidate_indexes.append(feat)
            else: #feature specified in both rules
                if is_categorical: #if feature is categorical
                    crossover_canidate_indexes.remove(feat)
        crossover_canidate_indexes.sort()
        # Probabilistically apply crossover for each candidate feature based on feature type (quantitative or categorical)
        swap_probability = 0.5
        for feat in crossover_canidate_indexes:
            is_categorical = heros.env.feat_types[feat]
            if random.random() > swap_probability: #perform crossover at this feature position
                # Identify feature occurences between rules
                feature_occurence = 0
                if feat in o1_condition_indexes:
                    feature_occurence += 1
                if feat in o2_condition_indexes:
                    feature_occurence += 2
                # Perform exchange
                if feature_occurence == 1: #feature only in o1
                    rule_position = self.condition_indexes.index(feat)
                    other_rule.condition_values.append(self.condition_values.pop(rule_position))
                    other_rule.condition_indexes.append(feat)
                    self.condition_indexes.remove(feat)
                elif feature_occurence == 2: #feature only in o2
                    rule_position = other_rule.condition_indexes.index(feat)
                    self.condition_values.append(other_rule.condition_values.pop(rule_position))
                    self.condition_indexes.append(feat)
                    other_rule.condition_indexes.remove(feat)
                else: #feature in both o1 and o2
                    if not is_categorical: #if quantiative feature
                        feat_index_1 = self.condition_indexes.index(feat)
                        feat_index_2 = other_rule.condition_indexes.index(feat)
                        random_options = random.randint(0,3) #Picks one of four random options for performing quanatiative feature range crossover
                        if random_options == 0: # swap low range value
                            temp = self.condition_values[feat_index_1][0]
                            self.condition_values[feat_index_1][0] = other_rule.condition_values[feat_index_2][0]
                            other_rule.condition_values[feat_index_2][0] = temp
                        elif random_options == 1: #swap high range value
                            temp = self.condition_values[feat_index_1][1]
                            self.condition_values[feat_index_1][1] = other_rule.condition_values[feat_index_2][1]
                            other_rule.condition_values[feat_index_2][1] = temp
                        else: #merge condition values found in both rules to form a new range in only one of the rules
                            all_values = self.condition_values[feat_index_1] + other_rule.condition_values[feat_index_2]
                            new_min = min(all_values)
                            new_max = max(all_values)
                            if new_min < heros.env.feat_q_range[feat][0]: # if value range goes below that observed in training data, set low to negative infinity
                                new_min = -np.inf
                            if new_max > heros.env.feat_q_range[feat][1]: # if value range goes above that observed in training data, set high to positive infinity
                                new_max = np.inf
                            if random_options == 2: #merge feature value ranges into o1 and remove feature from o2
                                self.condition_values[feat_index_1] = [new_min,new_max]
                                other_rule.condition_values.pop(feat_index_2)
                                other_rule.condition_indexes.remove(feat)
                            else: #merge feature value ranges into o2 and remove feature from o1
                                other_rule.condition_values[feat_index_2] = [new_min,new_max]
                                self.condition_values.pop(feat_index_1)
                                self.condition_indexes.remove(feat)
        if heros.outcome_type == 'quant': #for quantitative outcomes, the outcome interval can also crossover
            if random.random() > swap_probability:
                random_options = random.randint(0,3) #Picks one of four random options for performing quanatiative outcome crossover (all ensure current outcome value within range)
                if random_options == 0: # swap low range value
                    temp = self.action[0]
                    self.action[0] = other_rule.action[0]
                    other_rule.action[0] = temp
                elif random_options == 1: #swap high range value
                    temp = self.action[1]
                    self.action[1] = other_rule.action[1]
                    other_rule.action[1] = temp
                elif random_options == 2: #merge outcome values found in both rules to form a new range used in both rules
                    all_values = self.action + other_rule.action
                    new_min = min(all_values)
                    new_max = max(all_values)
                    if new_min < heros.env.outcome_range[0]: # if value range goes below that observed in training data, set low to negative infinity
                        new_min = -np.inf
                    if new_max > heros.env.outcome_range[1]: # if value range goes above that observed in training data, set high to positive infinity
                        new_max = np.inf
                    self.action = [new_min,new_max]
                    other_rule.action = [new_min,new_max]
                else: # choose the narrower inner range for both rules
                    all_values = self.action + other_rule.action
                    all_values = sorted(all_values)
                    self.action = [all_values[1],all_values[2]]
                    other_rule.action = [all_values[1],all_values[2]]
                # Ensure min/max action range is in proper order
                self.action.sort()
                other_rule.action.sort()

    def quantitative_range_check_fix(self, values, state,random):
        """ Check a given values range [min,max] to ensure the full range of values is not captured. But if so, randomly limit the range. """
        if values[0] == -np.inf and values[1] == np.inf:
            swap_probability = 0.5
            if random.random() > swap_probability: 
                values[0] = state
            else:
                values[1] = state

    def mutation(self,instance_state,outcome_state,heros,random):
        """ Applies niche mutation to the condition of the rule -the resulting classifier will still match the current training instance.
            This form of mutation does not allow a specified categorical feature value to change (as the rule will no longer match).
            Includes three mutation options (specify, generalize, and mutate quantitative feature range).
            Each feature can only be the target of one mutation type so that mutations don't undo each other. """
        # Ensure rule has a specificity between 1 and rsl ***********************************
        if len(self.condition_indexes) == 0: #completely general rule
            self.specify_condition_covering(1,instance_state,heros,random)
        if len(self.condition_indexes) > heros.rsl: #rule more specific than rule specificity limit
            num_to_generalize = len(self.condition_indexes) - heros.rsl
            generalized_features = self.generalize_condition(num_to_generalize,[],random) #num_to_generalize, changed_features
        # Determine number of mutations to be made based on mutation rate ******************************************
        mutations_remaining = 0
        keep_going = True
        while keep_going:
            if random.random() < heros.mut_prob:
                mutations_remaining += 1
            else:
                keep_going = False
        if mutations_remaining > heros.env.num_feat: # In case of small number of features in dataset, but a hard limit on mutations
            mutations_remaining = heros.env.num_feat
        # Determine mutation option possibilities ******************************************************
        mutate_options = ['G','S','R'] #generalize feature, specify feature, and mutate quanatitative feature range
        changed_features = [] #tracks which features have already been mutated
        # Apply mutation(s) ***************************************************
        while mutations_remaining > 0:
            # Select mutation option
            possibility_list, quant_feat_list = self.get_mutate_possibilities(changed_features,mutations_remaining,heros)
            probability_list = [value / sum(possibility_list) for value in possibility_list]
            choice = random.choices(mutate_options, weights=probability_list, k=1)[0]
            # Apply selected mutation option
            if choice == 'G': #generalize a feature
                generalized_features = self.generalize_condition(1,changed_features,random)
                changed_features.append(generalized_features[0]) #generalized_features is a list of length 1
            elif choice == 'S': #specify a feature
                specified_features = self.specify_condition_mutation(1,instance_state,changed_features,heros,random)
                if len(specified_features) != 0:
                    changed_features.append(specified_features[0]) #specified_features is a list of length 1
                else: #no valid feature could be found to mutate to (due to exhausted changes or presence of missing values in instance)
                    pass
            else: #mutate quantitative feature range
                mutated_feature = self.mutate_quantitative_range(instance_state,quant_feat_list,heros,random)
                changed_features.append(mutated_feature)
            mutations_remaining -= 1
        #Perform check for value ranges that span all instances (matches everything) and adjust as needed
        for feat_index in range(len(self.condition_indexes)):
            if heros.env.feat_types[self.condition_indexes[feat_index]] == 0: #quantitative feature
                self.quantitative_range_check_fix(self.condition_values[feat_index], instance_state[self.condition_indexes[feat_index]],random)
        # Mutate any quantitative outcome
        if heros.outcome_type == 'quant': #for quantitative outcomes, the outcome interval can also mutate
            if random.random() < heros.mut_prob:
                swap_probability = 0.5
                outcome_range = heros.env.outcome_range[1] - heros.env.outcome_range[0]
                random_options = random.randint(0,2) #Picks one of three random options for performing quanatiative outcome mutation
                if random_options == 0 or random_options == 2: #mutate low value
                    range_radius = (outcome_range / 2.0)* random.randint(25,75)*0.01
                    if random.random() > swap_probability:
                        self.action[0] += range_radius
                    else:
                        self.action[0] -= range_radius
                if random_options == 1 or random_options == 2: #mutate high value
                    range_radius = (outcome_range / 2.0)* random.randint(25,75)*0.01
                    if random.random() > swap_probability:
                        self.action[1] += range_radius
                    else:
                        self.action[1] -= range_radius
                self.action.sort()
                if self.action[0] < heros.env.outcome_range[0]: # if value range goes below that observed in training data, set low to negative infinity
                    self.action[0] = -np.inf
                if self.action[1] > heros.env.outcome_range[1]: # if value range goes above that observed in training data, set high to positive infinity
                    self.action[1] = np.inf
                self.quantitative_range_check_fix(self.action,outcome_state,random)

    def get_mutate_possibilities(self,changed_features,mutations_remaining,heros):
        possibility_list = [None, None, None] #generalize, specify, range mutate
        current_specificity = len(self.condition_indexes) #tracks the rule specificity as it is mutated
        # Determine generalization possibilities **********************************************
        temp_feature_pool = copy.deepcopy(self.condition_indexes)
        # Remove features that have already been modified by mutation from consideration
        for feat in changed_features:
            if feat in temp_feature_pool:
                temp_feature_pool.remove(feat)
        modifiable_specificity = len(temp_feature_pool) #number of specified features that have not been changed
        protected_specificity = current_specificity - modifiable_specificity
        if current_specificity < 2 or modifiable_specificity == 0: #no generalization allowed
            possibility_list[0] = 0
        elif protected_specificity > 0: #any modifible specificity up to mutations remaining can be made
            possibility_list[0] = min([mutations_remaining, modifiable_specificity])
        else: #no protected specificity (make sure to preserve at least one specified feature)
            possibility_list[0] = min([mutations_remaining, current_specificity - 1])
        # Determing specification possibilities **********************************************
        temp_feature_pool = list(range(heros.env.num_feat))
        # Remove existing specified features from consideration
        for feat in self.condition_indexes:
            if feat in temp_feature_pool:
                temp_feature_pool.remove(feat)
        # Remove features that have already been modified by mutation from consideration
        for feat in changed_features:
            if feat in temp_feature_pool:
                temp_feature_pool.remove(feat)
        modifiable_generality = min([heros.rsl - current_specificity, len(temp_feature_pool)]) #max features to be specified
        if modifiable_generality == 0:
            possibility_list[1] = 0
        else:
            possibility_list[1] = min([mutations_remaining,modifiable_generality])
        # Determing muate range possibilities **********************************************
        # Identify any eligible quantitative features specified in rule
        quant_feat_list = []
        for feat in self.condition_indexes:
            if not feat in changed_features: #exclude features that have been mutated already
                if not heros.env.feat_types[feat]: #if quantitative
                    quant_feat_list.append(feat)
        # Update possibilities
        if len(quant_feat_list) >= mutations_remaining: #all range mutations allowed
            possibility_list[2] = mutations_remaining
        else: #limited number of range mutations available
            possibility_list[2] = len(quant_feat_list)
        return possibility_list, quant_feat_list
    
    def mutate_quantitative_range(self,instance_state,quant_feat_list,heros,random):
        """ Mutate the value range of a specified quantitative feature in a rule. 
            Based on Bacardit 2009 - Select one bound with uniform probability and add or subtract a randomly generated offset to bound, 
            of size between 0 and 50% of feature domain. Also ensure range still includes current instance feature value
        """
        changed = False
        while not changed and len(quant_feat_list) > 0:
            feat = random.sample(quant_feat_list,1)[0]
            quant_feat_list.remove(feat)
            if instance_state[feat] != None:
                global_feature_range = heros.env.feat_q_range[feat][1] - heros.env.feat_q_range[feat][1] 
                rule_position = self.condition_indexes.index(feat)
                mutate_range = random.random() * 0.5 * global_feature_range

                if random.random() > 0.5: #mutate low end
                    if random.random() > 0.5: #add to low end
                        self.condition_values[rule_position][0] += mutate_range
                    else: #subtract from low end
                        self.condition_values[rule_position][0] -= mutate_range
                else: #mutate high end
                    if random.random() > 0.5: #add to high end
                        self.condition_values[rule_position][1] += mutate_range
                    else: #subtract from high end
                        self.condition_values[rule_position][1] -= mutate_range
                #Repair range so low end specified first then high end.
                self.condition_values[rule_position].sort()
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
                changed = True
        return feat

    def specify_condition_covering(self,num_to_specify,instance_state,heros,random):
        """ Specifies rule features for covering. """
        if not heros.use_ek:
            # Identify candidate features for specification
            temp_feature_pool = list(range(heros.env.num_feat)) # For sampling feature indexes without replacement
            # Create list of features to specify
            while len(self.condition_indexes) < num_to_specify and len(temp_feature_pool) > 0: #Aims to ensure target number of specified features is met, and prevent completely general rules
                feat = random.sample(temp_feature_pool,1)[0]
                temp_feature_pool.remove(feat)
                if instance_state[feat] != None: 
                    self.condition_indexes.append(feat)
                    self.condition_values.append(self.set_condition_value(feat,instance_state,heros,random))
        else: #expert knowledge covering
            i = 0
            # Create list of features to specify
            while len(self.condition_indexes) < num_to_specify and i < heros.env.num_feat - 1: #Aims to ensure target number of specified features is met, and prevent completely general rules
                feat = heros.env.ek_index_rank[i] #picks highest to lowest ranked features (based on expert knowledge scores)
                if instance_state[feat] != None: 
                    self.condition_indexes.append(feat)
                    self.condition_values.append(self.set_condition_value(feat,instance_state,heros,random))
                i += 1

    def specify_condition_mutation(self,num_to_specify,instance_state,changed_features,heros,random):
        """ Specifies rule features for mutation """
        # Identify candidate features for specification
        temp_feature_pool = list(range(heros.env.num_feat)) # For sampling feature indexes without replacement
        # Remove existing specified features from consideration
        for feat in self.condition_indexes:
            if feat in temp_feature_pool:
                temp_feature_pool.remove(feat)
        # Remove features that have already been modified by mutation from consideration
        for feat in changed_features:
            if feat in temp_feature_pool:
                temp_feature_pool.remove(feat)
        target_features = []
        while num_to_specify > 0 and len(temp_feature_pool) > 0: #Aims to ensure target number of specified features is met, and prevent completely general rules
            feat = random.sample(temp_feature_pool,1)[0]
            temp_feature_pool.remove(feat)
            if instance_state[feat] != None:
                self.condition_indexes.append(feat)
                self.condition_values.append(self.set_condition_value(feat,instance_state,heros,random))
                target_features.append(feat)
                num_to_specify -= 1
        return target_features

    def set_condition_value(self,feat,instance_state,heros,random):
        """ Used in 'covering' to set the rule-condition value (or value range for quantitative features) for a single specified feature in the new rule. """
        condition_value = None
        is_categorical = heros.env.feat_types[feat] # Assigned a value of 0 (False, Quantiative Feature) or 1 (True, Categorical Feature)
        # Categorical feature ******************************************************
        if is_categorical: 
            condition_value = instance_state[feat]
        # Quanatiative feature ******************************************************
        else:
            feature_range = heros.env.feat_q_range[feat][1] - heros.env.feat_q_range[feat][0]
            low = -np.inf
            high = np.inf
            while low == -np.inf and high == np.inf:
                range_radius = (feature_range / 2.0)* random.randint(25,75)*0.01
                low = instance_state[feat] - range_radius
                if low < heros.env.feat_q_range[feat][0]: # if value range goes below that observed in training data, set low to negative infinity
                    low = -np.inf
                high = instance_state[feat] + range_radius
                if high > heros.env.feat_q_range[feat][1]: # if value range goes above that observed in training data, set high to positive infinity
                    high = np.inf
                condition_value = [low,high]
        return condition_value
    
    def generalize_condition(self,num_to_generalize,changed_features,random):
        """ Generalizes rule features for mutation and ensuring rule specificity limit is maintained. """
        temp_feature_pool = copy.deepcopy(self.condition_indexes)
        for feat in changed_features:
            if feat in temp_feature_pool:
                #print("feat: "+str(feat))
                temp_feature_pool.remove(feat)
        target_features = random.sample(temp_feature_pool,num_to_generalize)
        for feat in target_features:
            rule_position = self.condition_indexes.index(feat)
            self.condition_indexes.remove(feat)
            self.condition_values.pop(rule_position)
        return target_features

    def complete_rule_evaluation_class(self,heros):
        """ Evaluates rule performance across the entire training dataset and update rule parameters accordingly. """
        # See if rule matches and if it's correct for each instance in the training data (updating match and correct counts)
        front_updated = False #only actively used by pareto front rule-fitness
        train_data = heros.env.train_data
        heros.env.num_instances
        self.match_set = []
        for instance_index in range(heros.env.num_instances):
            instance_state = train_data[0][instance_index] 
            outcome_state = train_data[1][instance_index]
            if self.match(instance_state,heros):
                self.match_cover += 1
                self.match_set.append(instance_state)
                if self.action == outcome_state:
                    self.correct_cover += 1
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
            # Check if new rule updates the rule pareto front
            front_updated = heros.rule_pareto.update_front(self.useful_accuracy,self.useful_coverage,['max','max']) 
            # Calculate and update the rule fitness
            self.fitness = heros.rule_pareto.get_pareto_fitness(self.useful_accuracy,self.useful_coverage, False,heros)
            if self.fitness is None: #Pareto front only has (0,0) for useful_accuracy and useful_coverage
                self.fitness = pow(self.accuracy, heros.nu)
            if front_updated: #all rule-fitnesses will be re-calculated externally
                return True
        else:
            print("Fitness metric not available.")
        return False
    
    def complete_rule_evaluation_quant(self,heros):
        """ Evaluates rule performance across the entire training dataset and update rule parameters accordingly. """
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
        if heros.fitness_function == 'accuracy':
            self.fitness = pow(self.useful_accuracy, heros.nu)
        elif heros.fitness_function == 'pareto':
            # Check if new rule updates the rule pareto front
            front_updated = heros.rule_pareto.update_front(self.useful_accuracy,self.useful_coverage,['max','max']) 
            # Calculate and update the rule fitness
            self.fitness = heros.rule_pareto.get_pareto_fitness(self.useful_accuracy,self.useful_coverage, False,heros)
            if self.fitness is None: #Pareto front only has (0,0) for useful_accuracy and useful_coverage
                self.fitness = pow(self.useful_accuracy, heros.nu)
            if front_updated: #all rule-fitnesses will be re-calculated externally
                return True
        else:
            print("Fitness metric not available.")
        return False
    

    def set_outcome_range_probability(self,heros):
        """ Calculate the probability that the quantitative outcome of a given instance in the training data will fall within the action range specified by this rule. """
        index = 0
        instance_count = 0
        while index < heros.env.num_instances and heros.env.outcome_ranked[index] <= self.action[1]:
            if heros.env.outcome_ranked[index] >= self.action[0]:
                instance_count += 1
            index += 1
        self.outcome_range_prob = instance_count/float(heros.env.num_instances)

    def update_rule_fitness(self,heros):
        """ Updates the rule fitness as a result of an update to the pareto front."""
        self.fitness = heros.rule_pareto.get_pareto_fitness(self.useful_accuracy,self.useful_coverage, False,heros)
        if self.fitness is None: #Pareto front only has (0,0) for useful_accuracy and useful_coverage
            self.fitness = pow(self.accuracy, heros.nu)

    def subsumes(self,other_rule,heros):
        """ Determines if 'self' rule meets conditions for subsuming the 'other_rule'. A rule is a subsumer if:
        (1) It has the same action as the other rule.
        (2) It has an accuracy >= the other rule.
        (3) It is more general than the other rule, covering all of the instance space of the other rule.
        """
        if heros.outcome_type == 'class':
            if not self.action == other_rule.action:
                return False
        elif heros.outcome_type == 'quant':
            if self.action[0] > other_rule.action[0] or self.action[1] < other_rule.action[1]:
                return False
        else:
            pass
        if not self.accuracy >= other_rule.accuracy:
            return False
        if not self.is_more_general(other_rule,heros):
            return False
        else:
            return True

    def is_more_general(self,other_rule,heros):
        """ Checks the conditions determining if self rule is more general than other rule as defined by being a candidate subsumer.
        (1) self has fewer specified features
        (2) other specifies at least the same subset of features as self
        (3) if a given feature is quanatiative the self_rule low boundary is <= than for other_rule
            and the self_rule high boundary is >= than for other_rule, i.e. the range is equal or larger (more general) at both ends."""
        if len(self.condition_indexes) >= len(other_rule.condition_indexes):
            return False #not more general
        for feat in self.condition_indexes: #for each feature specified in 'self' rule
            if feat not in other_rule.condition_indexes: #does self cover all the instance space of 'other_rule'
                return False #not more general
            if not heros.env.feat_types[feat]: #quantiative feature
                self_rule_position = self.condition_indexes.index(feat)
                other_rule_position = other_rule.condition_indexes.index(feat)
                # Current assumption - subsumer has a wider quanatiative feature range inclusive of the other rule's range
                if self.condition_values[self_rule_position][0] > other_rule.condition_values[other_rule_position][0]: #low end
                    return False #not more general
                if self.condition_values[self_rule_position][1] < other_rule.condition_values[other_rule_position][1]: #high end
                    return False #not more general
        return True

    def equals(self,other_rule):
        """ Checks the equivalence of two rules based on their conditions and actions. """
        if self.action == other_rule.action and len(self.condition_indexes) == len(other_rule.condition_indexes): #fast initial check of rule equality
            if sorted(self.condition_indexes) == sorted(other_rule.condition_indexes): #secondary check of rule equality (condition_indexes)
                for i in range(len(self.condition_indexes)): #final check of rule equality (condition_values)
                    other_rule_index = other_rule.condition_indexes.index(self.condition_indexes[i])
                    if not (self.condition_values[i] == other_rule.condition_values[other_rule_index]):
                        return False #final check yields inequality
                return True #rules are equivalent
        return False #initial or secondary checks yield inequality
    
    def get_deletion_vote(self,mean_fitness):
        """  Returns the vote for deletion of the rule. """

        """ EARLY IMPLEMENTATION
        if self.fitness == 0.0:
            deletion_vote = self.ave_match_set_size * self.numerosity * mean_fitness / (0.001 / self.numerosity)
        else: #regular calculation
            deletion_vote = self.ave_match_set_size * self.numerosity * mean_fitness / (self.fitness / self.numerosity) #Ryan -consider a different strategy
        return deletion_vote
        """
        if self.fitness == 0.0:
            deletion_vote = self.ave_match_set_size * self.numerosity / (0.001 / self.numerosity)
        else: #regular calculation
            deletion_vote = self.ave_match_set_size * self.numerosity / (self.fitness / self.numerosity) #Ryan -consider a different strategy
        return deletion_vote


    def update_ave_match_set_size(self, match_set_numerosity_sum,heros):
        """ Applies Widrow-Hoff algorithm to update average match set size in which this rule also matches. """
        self.ave_match_set_size = self.ave_match_set_size + heros.beta * (match_set_numerosity_sum - self.ave_match_set_size)
        
    """ ORIGINAL IMPLEMENTATION
    def update_ave_match_set_size(self, match_set_size, heros):
        " Applies Widrow-Hoff algorithm to update average match set size in which this rule also matches. "
        self.ave_match_set_size = self.ave_match_set_size + heros.beta * (match_set_size - self.ave_match_set_size)
    """

    def update_numerosity(self, num):
        """ Updates the numerosity of the classifier. """
        self.numerosity += num

    def show_rule(self):
        """ """
        print("Rule-------------------------------------------")
        print("ID: "+str(self.ID))
        print("Condition Indexes: "+str(self.condition_indexes))
        print("Condition Values: "+str(self.condition_values))
        print("Action: "+str(self.action))
        print("Fitness: "+str(self.fitness))
        print("Accuracy: "+str(self.accuracy))
        print("Match Cover: "+str(self.match_cover))
        print("Correct Cover: "+str(self.correct_cover))
        print("Birth Iteration: "+str(self.birth_iteration))
        print("Numerosity: "+str(self.numerosity))
        print("Ave Match Set Size: "+str(self.ave_match_set_size))
        print("Deletion Prob: "+str(self.deletion_prob))

    def show_rule_short(self,name):
        """ """
        print(str(name)+" Rule-------------------------------------------")
        print("Condition Indexes: "+str(self.condition_indexes))
        print("Condition Values: "+str(self.condition_values))