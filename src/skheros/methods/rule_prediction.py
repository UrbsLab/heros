import numpy as np

class RULE_PREDICTION():
    def __init__(self,heros,rule_population,random):
        """ Returns a prediction when applying the entire rule population to a given training or testing instance. """
        self.prediction = None
        self.covered = None
        # Classification outcome object
        self.prediction_proba = {}
        # Quantitative outcome object
        self.prediction_range = []
        self.majority_class_selection_made = False
        self.random_selection_made = False

        if heros.outcome_type == 'class': #classification outcome
            # Calculate vote sums based on rules in match set
            self.numerosity_sum = 0.0
            # Initialize prediction proba dictionary
            for current_class in heros.env.classes:
                self.prediction_proba[current_class] = 0.0
            # Calculate sum of each class's probabilities across all rules in the match set
            for rule_index in rule_population.match_set:
                rule = rule_population.pop_set[rule_index]
                for key in rule.instance_outcome_prop: #each possible class outcome
                    self.prediction_proba[key] += rule.instance_outcome_prop[key] * rule.numerosity
                self.numerosity_sum += rule.numerosity
            if not self.numerosity_sum == 0.0: #at least one matching rule
                # Calculate final prediction proba's - divide by total numerosity to get mean class probabilities across all matching rules (weighted by numerosity)
                for each in self.prediction_proba:
                    self.prediction_proba[each] /= self.numerosity_sum
                self.covered = True
            else: #no matching rules (leave prediction proba for all classes at 0.0)
                self.covered = False
            #Sort dictionary for consistency
            self.prediction_proba = dict(sorted(self.prediction_proba.items()))
            #Select prediction
            high_val = max(self.prediction_proba.values())
            candidate_actions = [k for k, v in self.prediction_proba.items() if v == high_val]
            if len(candidate_actions) == 1: #Best class prediction identified
                self.prediction = candidate_actions[0]
            else: # tie for best class (by default selects the majority class, or first listed class if no majority class)
                training_class_counts = []
                for each in candidate_actions: # each tied class
                    training_class_counts.append(heros.env.class_counts[each])
                high_class_count = max(training_class_counts)
                final_candidates_actions = []
                i = 0
                for each in candidate_actions:
                    if training_class_counts[i] == high_class_count:
                        final_candidates_actions.append(each)
                    i += 1
                if len(final_candidates_actions) == 1: #One of the tied classes is the majority class in the training data
                    self.prediction = final_candidates_actions[0]
                    self.majority_class_selection_made = True
                else: # There is still a tie (first listed class as final tiebreaker)
                    self.prediction = random.choice(final_candidates_actions)
                    self.random_selection_made = True
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
                #consider a default prediction of mean outcome for no matching instances (simplify above) 


    def get_prediction(self):
        """ Return outcome prediction made by rule population."""
        return self.prediction
    

    def get_prediction_proba_dictionary(self):
        """ Return prediction prediction for each class as a dictionary."""
        return self.prediction_proba
    

    def get_prediction_proba(self):
        """ Return prediction prediction for each class as a dictionary."""
        predict_proba_list = np.empty(len(sorted(self.prediction_proba.items())))
        counter = 0
        for k, v in sorted(self.prediction_proba.items()):
            predict_proba_list[counter] = v
            counter += 1
        return predict_proba_list
    

    def get_if_covered(self):
        """ Return prediction prediction for each class as a dictionary."""
        if self.covered:
            return 1 #instance was covered by at least one rule
        else:
            return 0 #instance was not covered by any rules


    def get_prediction_range(self):
        """ Return prediction prediction for each class as a dictionary."""
        return self.prediction_range
