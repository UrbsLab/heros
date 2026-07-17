import math
import copy


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


def Soft_Gini_SGD(self,instance_state, outcome_state, quant_feat_list,heros,random,np, k=1, learning_rate=1):
    """
    Calculate derivatives of gini with respect to lower and upper bounds for SGD.
    
    :param heros: The HEROS environment
    :param feature_index: The feature index to optimize
    :param k: Sigmoid steepness parameter
    :param learning_rate: The learning rate for gradient descent
    :return: Dictionary with updated lower and upper bounds

    Comtemplating adding a param for a limit on the decent, but not sure if it is necessary
    """

    def sigmoid(x):
        return 1 / (1 + math.exp(-x))
    
    def sigmoid_derivative(x):
        s = sigmoid(x)
        return s * (1 - s)


    #Selects a random quant feature to optimize
    feat = random.sample(quant_feat_list,1)[0]
    quant_feat_list.remove(feat)
    if instance_state[feat] != None:

        rule_lower = self.condition_values[feat][0]
        rule_upper = self.condition_values[feat][1]
        
        # Step 1: Calculate pi, N, and qc 

            #Call update gini 

        qc = self.soft_F[feat] / self.soft_N[feat]
        
        # Step 2: Calculate dpi/d(lower) and dpi/d(upper) for each instance
        dpi_d_lower = []
        dpi_d_upper = []
        
        for i, instance in enumerate(instance_state):
            feature_value = instance[feat]
            
            # dpi/d(lower) = k * σ'(k*(xi - lower)) * σ(k*(upper - xi))
            term1_lower = k * sigmoid_derivative(k * (feature_value - rule_lower))
            term2_lower = sigmoid(k * (rule_upper - feature_value))
            dpi_dlower = term1_lower * term2_lower
            dpi_d_lower.append(dpi_dlower)
            
            # dpi/d(upper) = σ(k*(xi - lower)) * (-k) * σ'(k*(upper - xi))
            term1_upper = sigmoid(k * (feature_value - rule_lower))
            term2_upper = -k * sigmoid_derivative(k * (rule_upper - feature_value))
            dpi_dupper = term1_upper * term2_upper
            dpi_d_upper.append(dpi_dupper)
        
        # Step 3: Calculate dqc/d(lower) and dqc/d(upper)
        # dqc/d(bound) = (1/N) * Σ(dpi/d(bound) * 1[yi = c]) - (qc/N) * Σ(dpi/d(bound))
        
        sum_dpi_lower = sum(dpi_d_lower)
        sum_dpi_upper = sum(dpi_d_upper)
        
        dqc_d_lower = {}
        dqc_d_upper = {}


        '''
        (1/self.soft_N[feat] * )
        '''
        
        for class_c in unique_classes:
            soft_correct_deriv_lower = sum(
                dpi_d_lower[i] for i in range(len(outcomes)) 
                if outcomes[i] == class_c
            )
            dqc_d_lower[class_c] = (soft_correct_deriv_lower / N) - (class_probs[class_c] / N) * sum_dpi_lower
            
            soft_correct_deriv_upper = sum(
                dpi_d_upper[i] for i in range(len(outcomes)) 
                if outcomes[i] == class_c
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

        ###################### DO I ADD SOMETHING HERE TO LIMIT THE DECENT (ESPECIALLY IF IT IS TOO LARGE OR SMALL)? ######################

        if abs(d_gini_d_lower) > 1e-5:
            rule_lower -= learning_rate * d_gini_d_lower
            self.condition_values[feat][0] = rule_lower
        else:
            return feat


        if abs(d_gini_d_upper) > 1e-5:
            rule_upper -= learning_rate * d_gini_d_upper
            self.condition_values[feat][1] = rule_upper
        else:
            return feat

        
    return feat

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


def delta_rule_mutation(self,instance_state, outcome_state, quant_feat_list,random,lr=0.01):
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
                # contract slightly toward this instance (tighten around confirmed-correct region)
                if dist_lower < dist_upper:
                    lower += lr * dist_lower * 0.1  # nudge lower up slightly, staying below x
                else:
                    upper -= lr * dist_upper * 0.1
            else:
                # this instance is incorrectly matched — push the nearer boundary away from it
                if dist_lower < dist_upper:
                    lower -= lr * (1 - dist_lower)  # push lower boundary down, excluding x more
                else:
                    upper += lr * (1 - dist_upper)

            self.condition_values[rule_position] = [lower, upper]

    return feat

