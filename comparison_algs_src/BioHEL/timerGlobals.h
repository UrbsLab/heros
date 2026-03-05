#ifndef _TIMER_GLOBAL_H_
#define _TIMER_GLOBAL_H_

#include "timingProcess.h"

class timerGlobals: public timingProcess {
public:
	int minClassifiersInit;
	int maxClassifiersInit;
	int penalizeMin;
	int ignoreMissingValues;
	int numClasses;
	int defaultClass;
	int defaultClassPolicy;
	int elitismEnabled;
	int smartInit;
	double probOne;
	int numAttributes;
	int numAttributesMC;
	int doTrainAndClean;
	double cleanProb;
	double generalizingProb;
	int numRepetitionsLearning;


	timerGlobals();
	void initialize(populationWrapper *pPW) {pw=pPW;}
	void newIteration(int iteration,int finalIteration);
	void dumpStats(int iteration);
};

#endif
