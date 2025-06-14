import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
#from sklearn.preprocessing import MinMaxScaler

class FEAT_TRACK:
    def __init__(self,heros):
        self.ft_scores = np.array([[0.0]*heros.env.num_feat for i in range(heros.env.num_instances)]) #feature tracking score array

    #ideas: divide rule scores by the number of features specified in rule (discribute credit for the accuracy)
    def get_correct_set_scores(self,heros):
        """ Generate a feature score list based on the correct set of the current instance"""
        correct_score_list = np.array([0.0]*heros.env.num_feat)
        correct_rule_count = 0
        for rule_index in heros.rule_population.correct_set:
            correct_rule_count += heros.rule_population.pop_set[rule_index].numerosity
            specificity = len(heros.rule_population.pop_set[rule_index].condition_indexes)
            for feat in heros.rule_population.pop_set[rule_index].condition_indexes:
                #correct_score_list[feat] += heros.rule_population.pop_set[rule_index].useful_accuracy * heros.rule_population.pop_set[rule_index].numerosity / specificity\
                #correct_score_list[feat] += heros.rule_population.pop_set[rule_index].fitness
                correct_score_list[feat] += heros.rule_population.pop_set[rule_index].useful_accuracy
        return correct_score_list, correct_rule_count

    def get_incorrect_set_scores(self,heros):
        """ Generate a feature score list based on the incorrect set of the current instance"""
        incorrect_score_list = np.array([0.0]*heros.env.num_feat)
        incorrect_rule_count = 0
        for rule_index in heros.rule_population.incorrect_set:
            incorrect_rule_count += heros.rule_population.pop_set[rule_index].numerosity
            specificity = len(heros.rule_population.pop_set[rule_index].condition_indexes)
            for feat in heros.rule_population.pop_set[rule_index].condition_indexes:
                #incorrect_score_list[feat] += heros.rule_population.pop_set[rule_index].useful_accuracy * heros.rule_population.pop_set[rule_index].numerosity / specificity
                incorrect_score_list[feat] += heros.rule_population.pop_set[rule_index].useful_accuracy
        return incorrect_score_list, incorrect_rule_count

    def update_ft_scores_new(self,outcome_prediction,action,heros):
        """ Updates feature tracking scores for current instance. """
        # Make correct and incorrect set score lists
        correct_score_list, correct_rule_count = self.get_correct_set_scores(heros)
        correct_score_list = correct_score_list / correct_rule_count #scale list by numerosity weighted number of correct rules
        if outcome_prediction != action: #incorrect prediction
            incorrect_score_list, incorrect_rule_count = self.get_incorrect_set_scores(heros)
            if incorrect_rule_count != 0:
                incorrect_score_list = incorrect_score_list / incorrect_rule_count #scale list by numerosity weighted number of correct rules
            else:
                incorrect_score_list = [0.0]*heros.env.num_feat
        # Update feature tracking scores for this instance
        for feat in range(heros.env.num_feat):
            if outcome_prediction == action: #correct prediction made
                self.ft_scores[heros.env.instance_index][feat] += correct_score_list[feat]
            else: #incorrect prediction made
                update = correct_score_list[feat] - incorrect_score_list[feat]
                if update < 0:
                    update = 0
                self.ft_scores[heros.env.instance_index][feat] += update

    def update_ft_scores(self,outcome_prediction,action,heros):
        """ Updates feature tracking scores for current instance. """
        # Make correct set score lists
        correct_score_list, correct_rule_count = self.get_correct_set_scores(heros)
        # Update feature tracking scores for this instance
        for feat in range(heros.env.num_feat):
            self.ft_scores[heros.env.instance_index][feat] += correct_score_list[feat]

    def update_ft_scores_wh(self,outcome_prediction,action,heros):
        """ Updates feature tracking scores for current instance. """
        # Make correct set score lists
        correct_score_list, correct_rule_count = self.get_correct_set_scores(heros)
        # Update feature tracking scores for this instance
        for feat in range(heros.env.num_feat):
            self.ft_scores[heros.env.instance_index][feat] += heros.beta * (correct_score_list[feat] - self.ft_scores[heros.env.instance_index][feat])

    def batch_calculate_ft_scores(self,heros):
        """ Calculates feature tracking scores of all instances in the training dataset based on a fixed final rule population. """
        for instance_index in range(heros.env.num_instances):
            # Get current instance properties
            instance_state = heros.env.train_data[0][instance_index]
            outcome_state = heros.env.train_data[1][instance_index]
            # Make {M}
            heros.rule_population.make_eval_match_set(instance_state,heros)
            # Make {C}
            heros.rule_population.make_correct_set(outcome_state,heros)
            # Make correct set score lists
            correct_score_list, correct_rule_count = self.get_correct_set_scores(heros)
            # Set feature tracking scores for this instance
            for feat in range(heros.env.num_feat):
                self.ft_scores[instance_index][feat] += correct_score_list[feat]
            heros.rule_population.clear_sets()

    def export_ft_scores(self,heros):
        """ Prepares and exports a dataframe capturing the feature tracking scores."""
        columns = list(range(heros.env.num_feat))
        ft_df = pd.DataFrame(self.ft_scores, columns=columns)
        ft_df['id'] = heros.env.instance_ids
        return ft_df 
    
    def plot_clustered_ft_heatmap(self, ft_df, feature_names, show=True, save=False, output_path=None, data_name=None):
        """ Generates a clustered heatmap of feature tracking scores. Scores values are scaled within rows between 0 and 1 for 
            hierarchical clustering between both rows and columns to determine clustering arrangment in plot. Scores values are 
            globally scaled (min/max out of all feature tracking scores) for displaying heatmap values on plot. 
            This makes clustering based on relative ft scores within an instance, while preserving relative feature tracking
            scores between individual instances to better identify instances with lower overall tracking scores.
        """
        # Prepare dataframe
        ft_df.columns = feature_names #Add original feature names back to columns rather than indexes
        # Scale feature tracking scores
        df_min = ft_df.min().min()
        df_max = ft_df.max().max()
        ft_df_all_scaled = (ft_df - df_min) / (df_max - df_min)
        ft_df_row_scaled = ft_df.apply(lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else np.zeros_like(x), axis=1) # Scale within rows individually
        # Perform hierarchical clustering on columns
        col_linkage = linkage(ft_df_row_scaled.T, method='average', metric='euclidean', optimal_ordering=True)
        # Perform hierarchical clustering on rows
        row_linkage = linkage(ft_df_row_scaled, method='average', metric='euclidean', optimal_ordering=True)
        # Create a seaborn clustermap
        clustermap = sns.clustermap(ft_df_all_scaled, row_linkage=row_linkage, col_linkage=col_linkage, cmap='viridis', figsize=(10, 10))
        clustermap.ax_heatmap.set_xlabel('Features', fontsize=12)
        clustermap.ax_heatmap.set_ylabel('Instances', fontsize=12)
        clustermap.ax_heatmap.set_yticks([])
        # Dynamicaly update x-tick label text size based on number of features in the dataset (up to a minimum )
        num_features = ft_df_all_scaled.shape[1]
        min_text_size = 4
        max_text_size = 12
        font_size = max(min_text_size, max_text_size - num_features // min_text_size)  # Adjust font size based on the number of features
        clustermap.ax_heatmap.set_xticklabels(clustermap.ax_heatmap.get_xticklabels(), rotation=90, fontsize=font_size)
        if save:
            plt.savefig(output_path+'/'+data_name+'_clustered_ft_heatmap.png', bbox_inches="tight")
        if show:
            plt.show()
