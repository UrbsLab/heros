#include "ga.h"
#include "configManagement.h"
#include "timerGlobals.h"
#include "random.h"
#include "timeManagement.h"
#include "messageBuffer.h"
#include "timerGlobals.h"
#include "attributesInfo.h"

extern Random rnd;
extern timeManagement tm;
extern instanceSet *is;
extern configManagement cm;
static int optimizationMethod;
extern int lastIteration;
extern messageBuffer mb;
extern int numTasks;
extern timerGlobals *tGlobals;


int rankOrder(const void *pA, const void *pB) {
	rank *a = (rank *) pA;
	rank *b = (rank *) pB;

	return a->ind->compareToIndividual2(b->ind, optimizationMethod);
}

void geneticAlgorithm::createPopulationRank() {
	int i;

	for (i = 0; i < popSize; i++) {
		populationRank[i].pos = i;
		populationRank[i].ind = population[i];
	}
	qsort(populationRank, popSize, sizeof(rank), rankOrder);
}

void geneticAlgorithm::initializePopulation() {
	int i;

	popSize = (int) cm.getParameter(POP_SIZE);

	population = new classifier *[popSize];
	offspringPopulation = new classifier *[popSize];
	if (!population || !offspringPopulation) {
		perror("out of memory");
		exit(1);
	}

	for (i = 0; i < popSize; i++) {
		population[i] = cf->createClassifier();
		if (!population[i]) {
			perror("out of memory");
			exit(1);
		}
	}

	populationRank = new rank[popSize];
	flagResetBest = 0;
	currentIteration = 0;
}

void geneticAlgorithm::initializeBalancedPopulation() {
	int i;

	popSize = (int) cm.getParameter(POP_SIZE);

	population = new classifier *[popSize];
	offspringPopulation = new classifier *[popSize];
	if (!population || !offspringPopulation) {
		perror("out of memory");
		exit(1);
	}

	//Add just one classifier with 0 atts
	i = 0;
	population[i++] = cf->createClassifier(0);

	//Add twice the number of atts of classifiers with 1 atts
	for (; i < tGlobals->numAttributesMC * 2 + 1; i++) {
		population[i] = cf->createClassifier(1);
	}

	int tam;
	for (; i < popSize; i++) {
		tam = i % (tGlobals->numAttributesMC - 2);
		population[i] = cf->createClassifier(tam + 2);
		//cout << tam << " " << i << "\n";
		if (!population[i]) {
			perror("out of memory");
			exit(1);
		}
	}

	populationRank = new rank[popSize];
	flagResetBest = 0;
	currentIteration = 0;
}

void geneticAlgorithm::doFitnessComputations() {
	int i;

	classifier ** pop = population;


	//printf("Fitness serial\n");
	for (i = 0; i < popSize; i++) {
		pop[i]->fitnessComputation();
	}

//	for (i = 1; i < popSize; i++) {
//		double fitness = population[i]->getFitness();
//		if (fitness > maxFitness) {
//			maxFitness = fitness;
//		}
//		if (fitness < minFitness) {
//			minFitness = fitness;
//		}
//	}

}

void geneticAlgorithm::resetBest() {
	flagResetBest = 1;
}

void geneticAlgorithm::checkBestIndividual() {
	int i;

	int currVer = is->getCurrentVersion();

	if (best[currVer] == NULL) {
		best[currVer] = cf->cloneClassifier(populationRank[0].ind);
	} else {

		best[currVer]->fitnessComputation();
		if (best[currVer]->compareToIndividual(populationRank[0].ind,
				optimizationMethod) < 0) {
			//mb.printf("Best indiv %d replaced\n",currVer);
			cf->deleteClassifier(best[currVer]);
			best[currVer] = cf->cloneClassifier(populationRank[0].ind);
		}
	}
}

void geneticAlgorithm::destroyPopulation() {
	int i;

	for (i = 0; i < popSize; i++)
		cf->deleteClassifier(population[i]);
	delete population;
	delete populationRank;
	delete offspringPopulation;
}

geneticAlgorithm::geneticAlgorithm(classifierFactory *pCF, int balanced) {
	cf = pCF;

	optimizationMethod = (int) cm.getParameter(MAX_MIN);

	numVersions = is->numVersions();
	best = new classifier *[numVersions + 1];
	for (int i = 0; i < numVersions; i++)
		best[i] = NULL;

	if (balanced) {
		initializeBalancedPopulation();
	} else {
		initializePopulation();
	}
	doFitnessComputations();
	createPopulationRank();
	checkBestIndividual();
}

geneticAlgorithm::~geneticAlgorithm() {
	destroyPopulation();
	for (int i = 0; i < numVersions; i++)
		if (best[i])
			cf->deleteClassifier(best[i]);
	delete best;
}


