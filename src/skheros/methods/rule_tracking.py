import pandas as pd
import matplotlib.pyplot as plt

class RULE_TRACK:
    def __init__(self, heros):
        """ """
        self.correct_predict_list = [] #list capturing whether or not (1 vs. 0) an accurate prediction was made on the last 'n' instances (n based on prediction_window) - classification
        self.error_predict_list = [] #list capturing the absolute error of the predictions made on the last 'n' instances (n based on prediction_window) - regression
        self.prediction_window = 50 #default/minimum prediction window
        if heros.track_performance > self.prediction_window:
            self.prediction_window = heros.track_performance
        if heros.outcome_type == 'class': #class outcome
            self.tracking_header = ["Iteration",
                            "Pred.Acc.Est.",
                            "Unique Rule Count",
                            "Rule Pop Size",
                            "Total Time"]
        elif heros.outcome_type == 'quant':
            self.tracking_header = ["Iteration",
                            "Pred.Error.Est.",
                            "Unique Rule Count",
                            "Rule Pop Size",
                            "Total Time"]
        else:
            pass
        self.tracking_list = []
        self.tracking_entry = []


    def update_prediction_list(self,outcome_prediction,outcome_truth,heros):
        """ Updates a windowed list of the most recent instance predictions (i.e. whether those predictions were correct or not)"""
        if heros.outcome_type == 'class': #class outcome
            if len(self.correct_predict_list) == self.prediction_window:
                del self.correct_predict_list[0]
            if outcome_prediction == outcome_truth:
                self.correct_predict_list.append(1)
            else:
                self.correct_predict_list.append(0)
        elif heros.outcome_type == 'quant': #quantitative outcome
            if len(self.error_predict_list) == self.prediction_window:
                del self.error_predict_list[0]
            self.error_predict_list.append(abs(outcome_prediction - outcome_truth))
        else:
            pass


    def update_performance_tracking(self,iteration,heros):
        """ Adds algorithm performance tracking information for the current training iteration."""
        # Update current global time
        heros.timer.update_global_time() 
        if heros.outcome_type == 'class': #class outcome
            # Update windowed rule-population prediction accuracy estimate
            if len(self.correct_predict_list) != 0:
                window_performance = sum(self.correct_predict_list)/len(self.correct_predict_list)
            else:
                window_performance = None
        elif heros.outcome_type == 'quant':
            # Update windowed rule-population prediction error estimate
            if len(self.error_predict_list) != 0:
                window_performance = sum(self.error_predict_list)/len(self.error_predict_list)
            else:
                window_performance = None
        else:
            pass
        # Create tracking entry
        self.tracking_entry = [iteration+1,
                          window_performance,
                          len(heros.rule_population.pop_set),
                          heros.rule_population.micro_pop_count,
                          heros.timer.time_global]
        # Add new tracking entry to the tracking list
        self.tracking_list.append(self.tracking_entry)


    def get_performance_tracking_df(self):
        """ Returns performance tracking over all training iterations as a dataframe. """
        tracking_df = pd.DataFrame(self.tracking_list,columns=self.tracking_header)
        return tracking_df
    

    def print_tracking_entry(self):
        """ Prints the tracking information for the current training iteration. """
        self.tracking_entry = [round(num,3) for num in self.tracking_entry]
        report_df = pd.DataFrame([self.tracking_entry], columns=self.tracking_header,index=None)
        print(report_df)


    def plot_rule_tracking(self,show=True, save=False, output_path=None):
        tracking_df = self.get_performance_tracking_df()
        # Plot Windowed Prediction Accuracy Estimates of HEROS Phase 1 Rule Population across learning iterations
        fig, ax1 = plt.subplots()
        ax1.plot(tracking_df['Iteration'], tracking_df["Pred.Acc.Est."], label='Balanced Prediction Accuracy Estimate', color='blue')
        ax1.set_ylabel('Window-Estimated Training Accuracy', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        # Create a second y-axis sharing the same x-axis
        ax2 = ax1.twinx()
        ax2.plot(tracking_df['Iteration'], tracking_df['Unique Rule Count'], label="Unique Rule Count", color='red')
        ax2.set_ylabel('Macro Population Size (Unique Rule Count)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        if save:
            plt.savefig(output_path+'/rule_tracking_line_graph.png', bbox_inches="tight")
        if show:
            plt.show()