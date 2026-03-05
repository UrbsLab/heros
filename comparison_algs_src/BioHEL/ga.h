#ifndef _GA_
#define _GA_

#include "factory.h"
#include "random.h"
#include "instanceSet.h"
#include <stdlib.h>
#include <stdio.h>

typedef struct {
        int pos;
        classifier *ind;
} rank;

extern instanceSet *is;


class geneticAlgorithm {
	int currentIteration;
	int popSize;
	classifierFactory *cf;
	classifier **population, **offspringPopulation;
	rank *populationRank;
	int flagResetBest;
	int numVersions;
	classifier **best;


	void checkBestIndividual();



	void initializePopulation();
         void initializeBalancedPopulation();

#include "crossover.h"
#include "scaling.h"
#include "replacement.h"
#include "selection.h"
#include "mutation.h"

public:
	void doFitnessComputations();
	void destroyPopulation();
    geneticAlgorithm(classifierFactory *cf,int balanced=0);
	~geneticAlgorithm();
	void doIterations(int num);
	classifier **getPopulation() { return population; }
	classifier *getBest() { return best[is->getCurrentVersion()]; }
	classifier *getWorst() {
		return population[populationRank[popSize - 1].pos];
	}
	rank *getPopulationRank() { return populationRank; }
	void resetBest();
		void createPopulationRank();
};

#endif
