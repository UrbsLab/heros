#include "populationWrapper.h"
#include <stdio.h>

populationWrapper::populationWrapper(int pPopSize, int balanced)
{
	cf=new classifierFactory;
        ga=new geneticAlgorithm(cf, balanced);
	popSize = pPopSize;
}

populationWrapper::~populationWrapper()
{
        if (ga != NULL) delete ga;
        if (cf != NULL) delete cf;
}

void populationWrapper::activateModifiedFlag()
{
	int i;

	rank *rk=ga->getPopulationRank();
	for(i=0;i<popSize;i++) rk[i].ind->activateModified();
	ga->resetBest();
}

rank *populationWrapper::getPopulationRank()
{
	int i;

	return ga->getPopulationRank();
}

void populationWrapper::doFitnessCalculations() {
	ga->doFitnessComputations();
}

void populationWrapper::createPopulationRank() {
    ga->createPopulationRank();
}

void populationWrapper::gaIteration()
{
	ga->doIterations(1);
}

void populationWrapper::releasePopulation()
{
	ga->destroyPopulation();
}

classifier *populationWrapper::getBestOverall()
{
	return (classifier *)ga->getBest();
}

classifier **populationWrapper::getPopulation()
{
	return (classifier **)ga->getPopulation();
}


classifier *populationWrapper::getBestPopulation()
{
	rank *rk=ga->getPopulationRank();
	return (classifier *)rk[0].ind;
}

classifier *populationWrapper::getWorstPopulation()
{
	rank *rk=ga->getPopulationRank();
	return (classifier *)rk[popSize - 1].ind;
}

double populationWrapper::getAverageLength()
{
	int i;
	double ave=0;

	rank *rk=ga->getPopulationRank();

	for(i=0;i<popSize;i++) ave+=rk[i].ind->getLength();

	return ave/(double)popSize;
}

void populationWrapper::getAverageDevAccuracy(double &ave,double &dev)
{
	int i;
	ave=0;
	dev=0;

	rank *rk=ga->getPopulationRank();

	for(i=0;i<popSize;i++) {
		double acc=((classifier *)rk[i].ind)->getAccuracy();
		ave+=acc;
		dev+=(acc*acc);
	}
	dev-=(ave*ave)/(double)popSize;
	dev/=(double)(popSize-1);
	dev=sqrt(dev);
	ave/=(double)popSize;
}

void populationWrapper::getAverageAccuracies(double &ave1,double &ave2)
{
	int i;
	ave1=0;
	ave2=0;
	rank *rk=ga->getPopulationRank();

	for(i=0;i<popSize;i++) {
		ave1+=((classifier *)rk[i].ind)->getAccuracy();
		ave2+=((classifier *)rk[i].ind)->getAccuracy2();
	}

	ave1/=(double)popSize;
	ave2/=(double)popSize;
}


double populationWrapper::getMaxAccuracy()
{
	int i;

	rank *rk=ga->getPopulationRank();
	double max=((classifier *)rk[0].ind)->getAccuracy();

	for(i=1;i<popSize;i++) {
		double percen=((classifier *)rk[i].ind)->getAccuracy();
		if(percen>max) max=percen;
	}

	return max;
}

classifier *populationWrapper::cloneClassifier(classifier *orig,int son)
{
	return cf->cloneClassifier(orig,son);
}

classifier *populationWrapper::createClassifier(int empty)
{
	return cf->createClassifier(empty);
}

void populationWrapper::destroyClassifier(classifier *orig)
{
	cf->deleteClassifier(orig);
}
