#include "timerCrossover.h"

extern timerCrossover *tCross;

extern "C" void calculateFitnessMixed(classifier ** population,
		instance ** instances, int * typeOfAttributes, int popSize,
		int numInstances, int strataOffset);

extern "C" void calculateFitnessReal(classifier ** population,
		instance ** instances, int popSize,
		int numInstances, int strataOffset);

void geneticAlgorithm::replacementAlgorithm() {
	totalReplacement();
	doFitnessComputations();
	createPopulationRank();
	if(tGlobals->elitismEnabled) doElitism();
}

void geneticAlgorithm::doElitism()
{
	int i,j;
	int n = 0;


#ifdef __CUDA_COMPILED__
	classifier ** pop = (classifier **) malloc(sizeof(classifier *) * numVersions);
	for (i = 0; i < numVersions; i++) {
		if (best[i]) {
			pop[n++] = best[i];

		}
	}

	instance ** instances = is->getInstancesOfIteration();
	if (ai.onlyRealValuedAttributes()) {
		calculateFitnessReal(pop, instances, n,
			is->getNumInstancesOfIteration(),
			is->getStrataOffsetOfIteration());
	} else {
		int * typesOfAttributes = ai.getTypeOfAttributes();
		calculateFitnessMixed(pop, instances, typesOfAttributes, n,
			is->getNumInstancesOfIteration(),
			is->getStrataOffsetOfIteration());
	}
#else
	for(i=0;i<numVersions;i++) {
		if(best[i]) {
			best[i]->fitnessComputation();
		}
	}
#endif

	int numV=numVersions;
	JVector<int> priorities(popSize+numV);
	for(i=0;i<popSize;i++) priorities.addElement(populationRank[i].pos);
	for(i=0;i<numV;i++) {
		if(best[i]) {
			int size=priorities.size();
			for(j=0;j<size;j++) {
				classifier *ind;
				int pos=priorities[j];
				if(pos>=popSize) {
					ind=best[pos-popSize];
				} else {
					ind=population[pos];
				}

				if(best[i]->compareToIndividual(ind,optimizationMethod)>0) {
					priorities.insertElementAt(popSize+i,j);
					break;
				}
			}
			if(j==size) {
				priorities.addElement(popSize+i);
			}
		}
	}

	JVector<int> elite;
	for(i=0;i<popSize;i++) {
		if(priorities[i]>=popSize) {
			//mb.printf("Elite element %d enters the population\n",priorities[i]-popSize);
			elite.addElement(priorities[i]-popSize);
		}
	}
	int index=0;
	int size=priorities.size();
	for(i=popSize;i<size;i++) {
		if(priorities[i]<popSize) {
			int pos=priorities[i];
			cf->deleteClassifier(population[pos]);
			population[pos]= cf->cloneClassifier(best[elite[index++]]);
		}
	}
	flagResetBest=0;
}


void geneticAlgorithm::totalReplacement()
{
	int i;

	for(i=0;i<popSize;i++) {
		cf->deleteClassifier(population[i]);
	}

	classifier **tempPop=population;
	population = offspringPopulation;
	offspringPopulation=tempPop;
}
