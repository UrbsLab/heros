#ifndef _TIMER_MUTATION_H_
#define _TIMER_Mutation_H_

#include "timingProcess.h"
#include "probabilityManagement.h"

class timerMutation: public timingProcess {
public:
	double mutationProb;

	timerMutation();
	void initialize(populationWrapper *pPW) {pw=pPW;}
	void newIteration(int iteration,int finalIteration);
	void dumpStats(int iteration);
};

#endif
