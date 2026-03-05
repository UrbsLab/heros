#ifndef _CLASSIFIER_H_
#define _CLASSIFIER_H_

#include "configCodes.h"
#include "attributesInfo.h"
#include "classifierFitness.h"
#include "timerHierar.h"
#include "timerGlobals.h"
#include "timerMDL.h"
#include "random.h"
#include <math.h>
#include <vector>

extern attributesInfo ai;
extern timerHierar *tHierar;
extern timerGlobals *tGlobals;
extern timerMDL *tMDL;
extern Random rnd;

class agentPerformanceTraining;

class classifier {
protected:
	int length; // Length of the individual in genes

	double scaledFitness;

	int front;
	double exceptionsLength; //For MDL fitness function
	double accuracy;
	double accuracy2;
	double coverage;
	double coverageTerm;
	double recall;
	double previousSpecificity;

	//for pruning purposes
	int numInstancesMatched;
	int numInstancesPos;
	int numInstancesPosOK;

	int numAttributesMC;
	double theoryLength;

public:

	int numAttributes;
	double fitness;
	int modif;

	inline classifier(char * string);

	inline classifier() {
		length = 0;
		modif = 1;
	}

	inline ~classifier() {
	}
	
	 virtual inline void boundsRemovalOperator(){}

    virtual inline int hasRequiredSpecificity()
    {
        return 0;
    }

	inline int getLength(void) {
		return length;
	}

	inline void fitnessComputation() {
		modif = 0;
		fitness = classifierFitness(*this);
	}

	inline int ** fitnessComputationWithMatchSet(int complete) {
		modif = 0;
		if(complete) {
			return classifierFitnessWithMatchSetComplete(*this);
		}
		return classifierFitnessWithMatchSet(*this);

	}

	inline void setScaledFitness(double pFitness) {
		scaledFitness = pFitness;
	}
	inline double getFitness(void) {
		return fitness;
	}
	inline void setFitness(double pFit) {
		fitness = pFit;
	}
	inline double getScaledFitness(void) {
		return scaledFitness;
	}
	inline void setAccuracy(double acc) {
		accuracy = acc;
	}
	inline double getAccuracy() {
		return accuracy;
	}
	inline void setAccuracy2(double acc) {
		accuracy2 = acc;
	}
	inline double getAccuracy2() {
		return accuracy2;
	}
	inline void setCoverage(double cov) {
		coverage = cov;
	}
	inline double getCoverage() {
		return coverage;
	}

	inline void adjustFitness() {
		if (tMDL->mdlAccuracy) {
			fitness = exceptionsLength;
		}
	}

	inline double getExceptionsLength() {
		return exceptionsLength;
	}
	inline void setExceptionsLength(double excep) {
		exceptionsLength = excep;
	}
	inline int isModified() {
		return modif;
	}
	inline void activateModified() {
		modif = 1;
	}
	inline double getTheoryLength() {
		return theoryLength;
	}

	inline double getCoverageTerm() {
		return coverageTerm;
	}

	inline void setCoverageTerm(double ct) {
		coverageTerm = ct;
	}

	inline double getRecall() {
		return recall;
	}

	inline void setRecall(double r) {
		recall = r;
	}

	inline double getPreviousSpecificity() {
		return previousSpecificity;
	}

	inline void setPreviousSpecificity(double d) {
		previousSpecificity = d;
	}

	virtual int getNumAttributesSpecified() {
	}

	virtual double getAttributesSpecificity() {
		return (double) 1.0 / (double) tGlobals->numAttributesMC;
	}
	virtual classifier *pruneAttributes() {
	}

        virtual classifier *pruneAttributes2(int index) {
	}

        virtual int cleanAttributes(int index) {
	}

        virtual int cleanAttributes2(int index) {
	}

	inline int getNumInstancesMatched() {
		return numInstancesMatched;
	}

	inline void setNumInstancesMatched(int m) {
		numInstancesMatched = m;
	}

	inline int getNumInstancesPos() {
		return numInstancesPos;
	}

	inline void setNumInstancesPos(int m) {
		numInstancesPos = m;
	}

	inline int getNumInstancesPosOK() {
		return numInstancesPosOK;
	}

	inline void setNumInstancesPosOK(int m) {
		numInstancesPosOK = m;
	}


	virtual int getClass() = 0;
	virtual int doMatch(instance * i) = 0;
	virtual void dumpPhenotype(char *string) = 0;
	virtual void dumpGenotype(char *string) {
	}
	virtual double computeTheoryLength() = 0;
	virtual void includeFrequncies(vector<vector<vector<int> > > * freq) {
	}

	// Second parent of CX & the two sons
	virtual void crossover(classifier *, classifier *, classifier *) = 0;

	virtual void mutation() = 0;
	virtual int numSpecialStages() = 0;
	virtual void doSpecialStage(int stage) = 0;
	
	    virtual int hasLocalSearch()
    {
        return 0;
    }
    virtual classifier* improveByLocalSearch()
    {
        return NULL;
    }
	
	virtual void postprocess() {
	}

	virtual void initiateEval() {
	}
	virtual void finalizeEval() {
	}

	virtual float calculateDistance(classifier * c) {
	}
	;

	virtual int equals(classifier * i2) {
	}

	inline int compareToIndividual(classifier * i2, int maxmin) {
		if (maxmin == MAXIMIZE) {
			if (fitness > i2->fitness)
				return +69;
			if (fitness < i2->fitness)
				return -69;
			return 0;
		}
		if (fitness < i2->fitness)
			return +69;
		if (fitness > i2->fitness)
			return -69;
		return 0;
	}

	inline int compareToIndividual2(classifier * i2, int maxmin) {
		if (maxmin == MAXIMIZE) {
			if (fitness > i2->fitness)
				return -69;
			if (fitness < i2->fitness)
				return +69;
			return 0;
		}
		if (fitness < i2->fitness)
			return -69;
		if (fitness > i2->fitness)
			return +69;
		return 0;
	}

};

#endif
