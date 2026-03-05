#include "timerHierar.h"
#include "messageBuffer.h"

extern messageBuffer mb;

timerHierar::timerHierar()
{
	if (!cm.thereIsParameter(HIERARCHICAL_SELECTION_ITERATION)) {
		enabled=0;
		activated=0;
		return;
	}
	enabled=1;

	useMDL=cm.thereIsParameter(HIERARCHICAL_SELECTION_USES_MDL);
	threshold = cm.getParameter(HIERARCHICAL_SELECTION_THRESHOLD);
	startIteration=(int)cm.getParameter(HIERARCHICAL_SELECTION_ITERATION);
	if(startIteration==0) activated=1;
	else activated=0;
}

void timerHierar::reinit()
{
	if(enabled) {
		if(startIteration==0) activated=1;
		else activated=0;
	}
}

void timerHierar::newIteration(int iteration,int lastIteration)
{
	if(!enabled) return;

	if(iteration==startIteration) {
		activated=1;
		mb.printf("Iteration %d:Hierarchical selection activated\n",iteration);
	}
}
