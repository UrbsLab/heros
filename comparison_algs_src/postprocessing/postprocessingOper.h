#ifndef POSTPROCESSINGOPER_H
#define POSTPROCESSINGOPER_H

void applyPostProcessing(classifier_aggregated & ind);

void ruleSetShake(classifier_aggregated & ind);
void attributeCleaning(classifier_aggregated & ind);
void attributeCleaning2(classifier_aggregated & ind);

void attributePrunning(classifier_aggregated & ind);
void attributePrunning2(classifier_aggregated & ind);

#endif // POSTPROCESSINGOPER_H
