#ifndef _POPULATION_WRAPPER_H_
#define _POPULATION_WRAPPER_H_

#include "ga.h"

class classifier;


class populationWrapper {
	geneticAlgorithm *ga;
	classifierFactory *cf;

public:
	int popSize;
        populationWrapper(int popSize, int balanced=0);
	~populationWrapper();
	void activateModifiedFlag();
	void gaIteration();
	void doFitnessCalculations();
        void createPopulationRank();
	void dumpPopulation();
	classifier *getBestOverall();
	classifier *getBestPopulation();
	classifier *getWorstPopulation();
	classifier **getPopulation();
	void releasePopulation();
	rank *getPopulationRank();
	double getAverageLength();
	double getAverageElements();
	double getAverageAliveElements();
	void getAverageDevAccuracy(double &ave,double &dev);
	void getAverageAccuracies(double &ave1,double &ave2);
	double getMaxAccuracy();
	classifier *cloneClassifier(classifier *orig,int son=0);
	void destroyClassifier(classifier *orig);
	classifier *createClassifier(int empty=0);
};

#endif
