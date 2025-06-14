class PREDICTION():
    def __init__(self,heros,rule_population):
        """ Returns a prediction when applying the entire rule population to a given training or testing instance. """
        self.prediction = None
        self.covered = None
        # Classification outcome object
        self.prediction_proba = {}
        # Quantitative outcome object
        self.prediction_range = []

        if heros.outcome_type == 'class': #classification outcome
            if len(rule_population.match_set) > 0: #at least one rule in match set (i.e. current instance is covered by the rule population)
                self.covered = True
                # Calculate vote sums based on rules in match set
                self.vote = {}
                for current_class in heros.env.classes:
                    self.vote[current_class] = 0.0
                for rule_index in rule_population.match_set:
                    rule = rule_population.pop_set[rule_index]
                    self.vote[rule.action] += rule.fitness * rule.numerosity #* heros.env.class_weights[rule.action]
                # Calculate prediction probabilities for each class
                proba_sum = 0
                for k,v in sorted(self.vote.items()):
                    self.prediction_proba[k] = v
                    proba_sum += v
                if proba_sum == 0: # prevent zero division (if match set empty)
                    for k,v in sorted(self.prediction_proba.items()):
                        self.prediction_proba[k] = 0
                else:
                    for k,v in sorted(self.prediction_proba.items()):
                        self.prediction_proba[k] = v/float(proba_sum)
                max_vote = 0.0
                best_class_list = []
                for current_class in heros.env.classes:
                    if self.vote[current_class] >= max_vote:
                        max_vote = self.vote[current_class]
                for current_class in heros.env.classes:
                    if self.vote[current_class] == max_vote:
                        best_class_list.append(current_class)
                # Assign predictions
                if len(best_class_list) > 1: #tie for best class
                    chosen_class = None
                    high_count = None
                    for current_class in best_class_list:
                        if chosen_class == None or heros.env.class_counts[current_class] > high_count: # > is most similar to how model operates using a decision threshold
                            chosen_class = current_class
                            high_count = heros.env.class_counts[current_class]
                    self.prediction = chosen_class
                else: #best class found
                    self.prediction = best_class_list[0]
            else: #empty match set (rule population does not cover current instance)
                self.prediction = heros.env.majority_class
                self.covered = False
        elif heros.outcome_type == 'quant': #quantitative outcome
            if len(rule_population.match_set) > 0: #at least one rule in match set (i.e. current instance is covered by the rule population)
                self.covered = True
                segment_range_list= [] # can include np.inf and -np.inf
                for rule_index in rule_population.match_set:
                    rule = rule_population.pop_set[rule_index]
                    low = rule.action[0]
                    if not low in segment_range_list:
                        segment_range_list.append(low)
                    high = rule.action[1]
                    if not high in segment_range_list:
                        segment_range_list.append(high)
                segment_range_list.sort()
                for i in range(0,len(segment_range_list)-1):
                    self.prediction_proba[(segment_range_list[i],segment_range_list[i+1])] = 0
                # Calculate votes for each segment range
                for rule_index in rule_population.match_set:
                    rule = rule_population.pop_set[rule_index]
                    low = rule.action[0]
                    high = rule.action[1]
                    for i in range(0,len(segment_range_list)-1):
                        if low <= segment_range_list[i] and high >= segment_range_list[i+1]: #does rule's outcome range include the current segment
                            self.prediction_proba[(segment_range_list[i],segment_range_list[i+1])] += rule.fitness * rule.numerosity
                # Identify the outcome rage with the strongest support
                self.prediction_range = max(self.prediction_proba,key=self.prediction_proba.get) #range given as a tuple
                #Find rules that overlap with this best range segment and gather their predictions and performance weights
                prediction_list = []
                weight_list = []
                numerosity_sum = 0
                for rule_index in rule_population.match_set:
                    rule = rule_population.pop_set[rule_index]
                    low = rule.action[0]
                    high = rule.action[1]
                    if low <= self.prediction_range[0] and high >= self.prediction_range[1]: #rule's outcome range includes the best range segment
                        prediction_list.append(rule.prediction * rule.numerosity)
                        weight_list.append(rule.fitness)
                        numerosity_sum += rule.numerosity
                weight_sum = sum(weight_list)
                if weight_sum == 0: #prediction is the average of all rule predictions (numerosity represents virtual copies of rules)
                    self.prediction = sum(prediction_list) / float(numerosity_sum)
                else: #prediction is the fitness weighted average of all predictions
                    prediction_sum = sum(a * b for a, b in zip(prediction_list, weight_list))
                    self.prediction = prediction_sum / (weight_sum * numerosity_sum)
            else: #empty match set (rule population does not cover current instance)
                self.prediction = sum(heros.env.outcome_ranked) / float(len(heros.env.outcome_ranked)) # Average of all training instance outcomes (default prediction)
                self.prediction_range = (self.prediction - heros.env.outcome_sd, self.prediction + heros.env.outcome_sd)
                self.prediction_proba = None
                self.covered = False

    def get_prediction(self):
        """ Return outcome prediction made by rule population."""
        return self.prediction
    
    def get_prediction_proba(self):
        """ Return prediction prediction for each class as a dictionary."""
        return self.prediction_proba

    def get_if_covered(self):
        """ Return prediction prediction for each class as a dictionary."""
        if self.covered:
            return 1 #instance was covered by at least one rule
        else:
            return 0 #instance was not covered by any rules

    def get_prediction_range(self):
        """ Return prediction prediction for each class as a dictionary."""
        return self.prediction_range
