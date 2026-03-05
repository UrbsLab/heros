#include <stdlib.h>
#include <stdio.h>
#include "random.h"
#include "timerEvolutionStats.h"
#include "instanceSet.h"

extern timerEvolutionStats *tEvolStats;
extern instanceSet *is;

void geneticAlgorithm::selectionAlgorithm()
{
	scalingAlgorithm();

	switch (selectionAlg) {
	case TOURNAMENT_WOR_SELECTION:
		TournamentSelectionWOR();
		break;
	case TOURNAMENT_SELECTION:
	default:
		TournamentSelection();
		break;
	}

	for (int i = 0; i < popSize; i++) cf->deleteClassifier(population[i]);
	classifier **tempPop=population;
	population = offspringPopulation;
	offspringPopulation=tempPop;
}

void geneticAlgorithm::TournamentSelectionWOR(void)
{
	int i, j, winner, candidate;

	Sampling samp(popSize);
	for (i = 0; i < popSize; i++) {
		//There can be only one
		winner=samp.getSample();
		for (j = 1; j < tournamentSize; j++) {
			candidate=samp.getSample();
			if(population[candidate]->
				compareToIndividual(population[winner],
				optimizationMethod)>0) {
				winner = candidate;
			}
		}
		offspringPopulation[i]=cf->cloneClassifier(population[winner]);
	}
}

void geneticAlgorithm::TournamentSelection(void)
{
	int i, j, winner, candidate;

	for (i = 0; i < popSize; i++) {
		//There can be only one
		winner=rnd(0,popSize-1);
		for (j = 1; j < tournamentSize; j++) {
			candidate=rnd(0,popSize-1);
			if(population[candidate]->
				compareToIndividual(population[winner],
					optimizationMethod)>0) {
				winner = candidate;
			}
		}
		offspringPopulation[i]=cf->cloneClassifier(population[winner]);
	}
}
