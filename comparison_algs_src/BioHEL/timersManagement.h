#ifndef _TIMERS_MANAGEMENT_H_
#define _TIMERS_MANAGEMENT_H_

#include "populationWrapper.h"
#include "JVector.h"
#include "timingProcess.h"

class timersManagement {
	JVector<timingProcess *> timers;
	int iteration;
public:
	timersManagement();
	~timersManagement();
	void incIteration(int lastIteration);
	void dumpStats(); 
	void reinit();
	void setPW(populationWrapper *pPW);
	void addTimer(timingProcess *tp){timers.addElement(tp);}
};

#endif
