#ifndef _TIMER_Discret_H_
#define _TIMER_Discret_H_

#include "timingProcess.h"

class timerSymbolicKR: public timingProcess {
	int enabled;
public:
	int *sizeAttribute;
	int *offsetAttribute;
	int ruleSize;
	double probSharp;

	timerSymbolicKR();
	void initialize(populationWrapper *pPW) {pw=pPW;}
	void newIteration(int iteration,int lastIteration) {}
	void dumpStats(int iteration);
};

#endif
