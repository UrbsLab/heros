#include "timersManagement.h"

#include "timerGlobals.h"
//#include "timerADI.h"
#include "timerMDL.h"
#include "timerHierar.h"
#include "timerMutation.h"
#include "timerSymbolicKR.h"
#include "timerRealKR.h"
#include "timerEvolutionStats.h"
#include "timerCrossover.h"

timerGlobals *tGlobals;
timerMDL *tMDL;
timerHierar *tHierar;
//timerADI *tADI;
timerRealKR *tReal;
timerSymbolicKR *tSymbolic;
timerMutation *tMut;
timerCrossover *tCross;
timerEvolutionStats *tEvolStats;

timersManagement::timersManagement()
{
	iteration = -1;

	tGlobals=new timerGlobals;
	//tADI=new timerADI;
	tMDL=new timerMDL;
	tHierar=new timerHierar;
	tReal=new timerRealKR;
	tSymbolic=new timerSymbolicKR;
	tMut=new timerMutation;
	tEvolStats=new timerEvolutionStats;
	tCross=new timerCrossover;

	addTimer(tGlobals);
	//addTimer(tADI);
	addTimer(tHierar);
	addTimer(tMDL);
	addTimer(tSymbolic);
	addTimer(tReal);
	addTimer(tMut);
	addTimer(tCross);
	addTimer(tEvolStats);
}

timersManagement::~timersManagement()
{
	delete tGlobals;
	//delete tADI;
	delete tHierar;
	delete tMDL;
	delete tSymbolic;
	delete tReal;
	delete tMut;
	delete tCross;
	delete tEvolStats;
}


void timersManagement::incIteration(int lastIteration)
{
	iteration++;

	int i;
	for(i=0;i<timers.size();i++)
		timers[i]->newIteration(iteration,lastIteration);
}

void timersManagement::reinit()
{
	iteration=-1;

	int i;
	for(i=0;i<timers.size();i++)
		timers[i]->reinit();
}


void timersManagement::dumpStats()
{
	int i;
	for(i=0;i<timers.size();i++)
		timers[i]->dumpStats(iteration);
}

void timersManagement::setPW(populationWrapper *pPW)
{
	int i;
	for(i=0;i<timers.size();i++)
		timers[i]->initialize(pPW);
}
