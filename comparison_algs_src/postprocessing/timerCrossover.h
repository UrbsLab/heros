#ifndef _TIMER_CROSSOVER_H_
#define _TIMER_CROSSOVER_H_

#include "timingProcess.h"

class timerCrossover: public timingProcess {
public:
	double crossoverProb;
	int cxOperator;

	int numBB;
	int *sizeBBs;
	int **defBBs;

	timerCrossover();
	void initialize(populationWrapper *pPW) {pw=pPW;}
	void newIteration(int iteration,int finalIteration);
	void dumpStats(int iteration);
};

#endif
