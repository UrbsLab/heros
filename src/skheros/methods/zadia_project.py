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


def Soft_Gini_SGD(self, heros, feature_index,np, random, k=1, learning_rate=0.01):
    """
    Calculate derivatives of gini with respect to lower and upper bounds.
    
    :param feature_index: The feature index to optimize
    :param k: Sigmoid steepness parameter
    :param learning_rate: Th -pe learning rate for gradient descent
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


        
    if feature_index not in self.condition_indexes:
        return None
    
    position = self.condition_indexes.index(feature_index)

    rule_lower = self.condition_values[position][0]
    rule_upper = self.condition_values[position][1]

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
        qc = soft_correct / N
        class_probs[class_c] = qc
    
    # Step 2: Calculate dpi/d(lower) and dpi/d(upper) for each instance
    dpi_d_lower = []
    dpi_d_upper = []
    
    for instance in filtered_states:
        feature_value = instance[feature_index]
        
        # dpi/d(lower) = -k * σ'(k*(xi - lower)) * σ(k*(upper - xi))
        term1_lower = -k * sigmoid_derivative(k * (feature_value - rule_lower))
        term2_lower = sigmoid(k * (rule_upper - feature_value))
        dpi_dlower = term1_lower * term2_lower
        dpi_d_lower.append(dpi_dlower)
        
        # dpi/d(upper) = σ(k*(xi - lower)) * (k) * σ'(k*(upper - xi))
        term1_upper = sigmoid(k * (feature_value - rule_lower))
        term2_upper = k * sigmoid_derivative(k * (rule_upper - feature_value))
        dpi_dupper = term1_upper * term2_upper
        dpi_d_upper.append(dpi_dupper)
    
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

    effective_lr = learning_rate * (heros.env.feat_q_range[feature_index][1]- heros.env.feat_q_range[feature_index][0])


    new_lower = rule_lower - effective_lr * d_gini_d_lower
    new_upper = rule_upper - effective_lr * d_gini_d_upper

    
    valid_instances = [instance_states[i] for i in valid_indices]

    for instance in valid_instances:
        if not new_lower < instance[feature_index] < new_upper:
            #Repair range to include current instance's feature value
            if abs(new_upper - instance[feature_index]) > abs(instance[feature_index] - new_lower): #instance value closer to low end
                new_lower = instance[feature_index]
            else:
                new_upper = instance[feature_index]    
    


    self.condition_values[position][0] = new_lower
    self.condition_values[position][1] = new_upper

    self.condition_values[position].sort()

    return [new_lower, new_upper]


    '''
    print (
        'new_lower:', rule_lower - learning_rate * d_gini_d_lower,
        ' new_upper', rule_upper - learning_rate * d_gini_d_upper,
        ' current_gini:', 1.0 - sum(qc ** 2 for qc in class_probs.values())
    )
    '''


def optimize_quantitative_range(self, instance_state, quant_feat_list,heros,random,np):
    """ Mutate the value range of a specified quantitative feature in a rule.
    """
    changed = False
    while not changed and len(quant_feat_list) > 0:
        feat = random.sample(quant_feat_list,1)[0]
        quant_feat_list.remove(feat)
        if instance_state[feat] != None:
            rule_position = self.condition_indexes.index(feat)

            new_bounds = Soft_Gini_SGD(self, heros, feat)

            if random.random() > 0.5: #mutate low end
                if random.random() > 0.5: #add to low end
                    self.condition_values[rule_position][0] = new_bounds["new_lower"]

            else: #mutate high end
                if random.random() > 0.5: #add to high end
                    self.condition_values[rule_position][1] = new_bounds["new_upper"]


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

