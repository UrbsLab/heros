#include "agentPerformanceTraining.h"
#include "timerGlobals.h"
#include "timerHierar.h"
#include "timerMDL.h"
#include "classifier.h"
#include "messageBuffer.h"

extern timerGlobals *tGlobals;
extern timerMDL *tMDL;
extern timerHierar *tHierar;
extern messageBuffer mb;

agentPerformanceTraining::agentPerformanceTraining(int pNumInstances,int pRuleClass)
{
	ruleClass=pRuleClass;

	numInstancesTotal = pNumInstances;
	numInstancesPosOK = 0;
	numInstancesMatched = 0;
	numInstancesPos = 0;
}

double agentPerformanceTraining::getFitness(classifier &ind)
{
	double fitness;

	if(tMDL->mdlAccuracy) {
		ind.computeTheoryLength();
		fitness=tMDL->mdlFitness(ind,this);
	} else {
		//fitness=2*(numInstancesTotal-numInstancesKO)+numInstancesOK; //+ind.getCoverageRatio();
		//fitness=getAccuracy2();
		fitness=getFMeasure();
		fitness*=fitness;
	}

	return fitness;
}
