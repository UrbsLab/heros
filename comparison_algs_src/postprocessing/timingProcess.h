#ifndef _TIMING_PROCESS_H_
#define _TIMING_PROCESS_H_

#include "configManagement.h"

extern configManagement cm;

class populationWrapper;

class timingProcess {
protected:
	populationWrapper *pw;
public:
	virtual void initialize(populationWrapper *pPW)=0;
	virtual void newIteration(int iteration,int finalIteration)=0;
	virtual void dumpStats(int iteration)=0;	
	virtual void reinit(){}
};

#endif
