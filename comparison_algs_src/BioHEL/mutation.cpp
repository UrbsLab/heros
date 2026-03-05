#include "timerMutation.h"

extern timerMutation *tMut;

void geneticAlgorithm::mutation()
{
	individualMutation();
	specialStages();
}

void geneticAlgorithm::specialStages()
{
	int i,j;
	int numStages=offspringPopulation[0]->numSpecialStages();
	for(i=0;i<numStages;i++) {
		for(j=0;j<popSize; j++) {
			offspringPopulation[j]->doSpecialStage(i);
		}
	}
}

void geneticAlgorithm::individualMutation()
{
	int i;
	for (i = 0; i < popSize; i++) {
		if(!rnd<tMut->mutationProb) {
			offspringPopulation[i]->mutation();
		}
	}
}

