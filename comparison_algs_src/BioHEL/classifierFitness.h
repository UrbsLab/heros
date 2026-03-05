#ifndef _CLASSIFIER_FITNESS_H_

#define _CLASSIFIER_FITNESS_H_

class classifier;
class instance;
class instanceSet;
class classifierFactory;
class classifier_aggregated;
#include "JVector.h"


double classifierFitness(classifier & ind);
void classifierBriefTest(classifier_aggregated &ind, instanceSet *is);
void classifierStats(classifier_aggregated &ind, char instancesFile[], const char *typeOfFile);
void evaluateIndividuals(classifierFactory *cf);
int isMajority(classifier & ind);

#endif
