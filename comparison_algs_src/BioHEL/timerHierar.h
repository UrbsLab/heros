#ifndef _TIMER_HIERAR_H_
#define _TIMER_HIERAR_H_

#include "timingProcess.h"

class timerHierar: public timingProcess {
	int startIteration;
	int enabled;
public:
	int useMDL;
	double threshold;
	double activated;

	timerHierar();
	void initialize(populationWrapper *pPW) {pw=pPW;}
	void newIteration(int iteration,int finalIteration);
	void dumpStats(int iteration) {}	
	void reinit();
};

#endif
