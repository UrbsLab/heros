#ifndef _CLASSIFIER_FITNESS_H_

#define _CLASSIFIER_FITNESS_H_

class classifier;
class instance;
class instanceSet;
class classifierFactory;
class classifier_aggregated;
#include "JVector.h"
#include "agentPerformance.h"

double classifierFitness(classifier & ind);

void classifierBriefTest(classifier_aggregated &ind, instanceSet *is);
void classifierStats(classifier_aggregated &ind, instanceSet *is, const char *typeOfFile);
void evaluateIndividuals(classifierFactory *cf);
int isMajority(classifier & ind);

int ** classifierFitnessWithMatchSet(classifier & ind);
int ** classifierFitnessWithMatchSetComplete(classifier & ind);

agentPerformance calculateAgentPerformance(classifier_aggregated & ind);

#endif
