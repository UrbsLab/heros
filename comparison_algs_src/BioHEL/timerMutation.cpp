#include "timerMutation.h"
#include <math.h>
#include "messageBuffer.h"

extern messageBuffer mb;

timerMutation::timerMutation()
{
	mutationProb = cm.getParameter(PROB_INDIVIDUAL_MUTATION);
}

void timerMutation::newIteration(int iteration,int lastIteration)
{
}

void timerMutation::dumpStats(int iteration)
{
}
