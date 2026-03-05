#include "timerCrossover.h"
#include "timerGlobals.h"
#include "sampling.h"

extern timerCrossover *tCross;
extern timerGlobals *tGlobals;

#define max(a,b) (a)>(b)?(a):(b)

void geneticAlgorithm::crossTwoParents(int parent1, int parent2, int son1,
				       int son2)
{
	offspringPopulation[son1] =
	    cf->cloneClassifier(population[parent1], 1);
	offspringPopulation[son2] =
	    cf->cloneClassifier(population[parent2], 1);

	population[parent1]->crossover(population[parent2]
				       , offspringPopulation[son1]
				       , offspringPopulation[son2]);
}

void geneticAlgorithm::crossOneParent(int parent, int son)
{
	offspringPopulation[son] =
	    cf->cloneClassifier(population[parent], 0);
}


void geneticAlgorithm::crossover()
{
	int i, j, k, countCross = 0;

	Sampling samp(popSize);
	int p1 = -1;
	for (j = 0; j < popSize; j++) {
		if (!rnd < tCross->crossoverProb) {
			if (p1 == -1) {
				p1 = samp.getSample();
			} else {
				int p2 = samp.getSample();
				crossTwoParents(p1, p2, countCross,
						countCross + 1);
				countCross += 2;
				p1 = -1;
			}
		} else {
			crossOneParent(samp.getSample(), countCross++);
		}
	}
	if (p1 != -1) {
		crossOneParent(p1, countCross++);
	}
}
