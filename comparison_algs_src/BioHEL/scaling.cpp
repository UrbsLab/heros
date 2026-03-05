#include <math.h>

void geneticAlgorithm::scalingAlgorithm() 
{
	int i;

	for(i=0;i<popSize;i++) {
		double value=population[i]->getFitness();
		value=identityScaling(value);
		population[i]->setScaledFitness(value);
	}
}

double geneticAlgorithm::identityScaling(double value)
{
	double res;

	if(optimizationMethod==MAXIMIZE) res=value;
	else res=maxFitness+minFitness-value;
		
	if(minFitness<0) res-=minFitness;
	
	return res;
}

