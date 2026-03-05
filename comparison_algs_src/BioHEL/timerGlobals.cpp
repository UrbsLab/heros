#include "instanceSet.h"
#include "timerGlobals.h"
#include "timerEvolutionStats.h"
#include "attributesInfo.h"
#include "messageBuffer.h"

extern messageBuffer mb;

extern attributesInfo ai;

timerGlobals::timerGlobals()
{
	minClassifiersInit =
		(int) cm.getParameter(INITIALIZATION_MIN_CLASSIFIERS);
	maxClassifiersInit =
		(int) cm.getParameter(INITIALIZATION_MAX_CLASSIFIERS);

	if (cm.thereIsParameter(IGNORE_MISSING_VALUES)) {
		ignoreMissingValues = 1;
	} else {
		ignoreMissingValues = 0;
	}

	if (cm.thereIsParameter(PENALIZE_MIN_SIZE)) {
		penalizeMin=(int)cm.getParameter(PENALIZE_MIN_SIZE);
	} else {
		penalizeMin=0;
	}

	defaultClassPolicy=(int)cm.getParameter(DEFAULT_CLASS);
	defaultClass=-1;
	switch(defaultClassPolicy) {
            case MAJOR:
                numClasses=ai.getNumClasses()-1;
                //defaultClass=0;
                defaultClass=ai.getMostFrequentClass();
                break;
            case MINOR:
                numClasses=ai.getNumClasses()-1;
                //defaultClass=0;
                defaultClass=ai.getLeastFrequentClass();
                break;
            case FIXED:
                numClasses=ai.getNumClasses()-1;
                defaultClass=(int)cm.getParameter(FIXED_DEFAULT_CLASS);
                break;
            case DISABLED:
                numClasses=ai.getNumClasses();
                break;
            case AUTO:
                numClasses=ai.getNumClasses()-1;
                break;

	}

	//printf("Default class %d\n",defaultClass);

	elitismEnabled=1;

	smartInit=cm.thereIsParameter(SMART_INIT);
	numAttributes=ai.getNumAttributes();
	numAttributesMC=numAttributes-1;

	doTrainAndClean=0;
	if(cm.thereIsParameter(RULE_CLEANING_PROB)) {
		doTrainAndClean=1;
		cleanProb=cm.getParameter(RULE_CLEANING_PROB);
	} else {
		cleanProb=0;
	}
	if(cm.thereIsParameter(RULE_GENERALIZING_PROB)) {
		doTrainAndClean=1;
		generalizingProb=cm.getParameter(RULE_GENERALIZING_PROB);
	} else {
		generalizingProb=0;
	}

	numRepetitionsLearning=(int)cm.getParameter(REPETITIONS_RULE_LEARNING);
}

void timerGlobals::newIteration(int iteration,int lastIt)
{
}

void timerGlobals::dumpStats(int iteration)
{
}
