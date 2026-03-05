#ifndef _AGENT_PERFORMANCE_TRAINING_H_
#define _AGENT_PERFORMANCE_TRAINING_H_

class classifier;

#include <math.h>

class agentPerformanceTraining {
	int ruleClass;

	//int numInstancesOK;
	//int numInstancesKO;
	//int numInstancesNC;

	int numInstancesPos;
	int numInstancesPosOK;
	int numInstancesTotal;
	int numInstancesMatched;

public:
	agentPerformanceTraining(int pNumInstances,int pRuleClass);

	inline void addMatch(int realClass,int predictedClass) {
		if(realClass==ruleClass) numInstancesPos++;
		numInstancesMatched++;

		if (predictedClass == realClass) {
			numInstancesPosOK++;
		}
	}

	inline void addNoMatch(int realClass) {
		if(realClass==ruleClass) numInstancesPos++;
	}

	inline double getAccuracy() { return (double)numInstancesPosOK/(double)numInstancesTotal; }
	//inline double getAccuracy() { return (double)numInstancesOK/(double)numInstancesTotal; }
	//inline double getError() { return (double)numInstancesKO/(double)numInstancesTotal; }
	//inline double getError2() {
	//	if(numInstancesOK+numInstancesKO==0) return 0;
	//	return (double)numInstancesKO/(double)(numInstancesOK+numInstancesKO);
	//}
	inline double getAccuracy2() {
		if(numInstancesMatched==0) return 0;
		return (double)numInstancesPosOK/(double)numInstancesMatched;
		//if(numInstancesOK+numInstancesKO==0) return 0;
		//return (double)numInstancesOK/(double)(numInstancesOK+numInstancesKO);
	}
	//inline double getCoverage() { return (double)(numInstancesOK+numInstancesKO)/(double)numInstancesTotal;}
	inline double getCoverage() { return (double)numInstancesMatched/(double)numInstancesTotal;}
    inline double getCoverage2() {  return (double)numInstancesPosOK/(double)numInstancesTotal;}
	inline int getNumOK() { return numInstancesPosOK;}
	inline int getNumPos() { return numInstancesPos;}
	inline int getNumMatched() { return numInstancesMatched; }
	inline int getNumKO() { return numInstancesMatched-numInstancesPosOK;}
	inline int getNumTotal() { return numInstancesTotal;}
	inline double getNC(){return (double)(1-numInstancesMatched)/(double)numInstancesTotal;}
	inline double getRecall(){ return (double)numInstancesPosOK/(double)numInstancesPos; }
	inline double getFMeasure() {
		double precision=getAccuracy2();
		double recall=getRecall();
		return 2*precision*recall/(precision+recall);
	}
	double getFitness(classifier &ind);

	inline void setNumMatched(int i) {
		numInstancesMatched = i;
	}

	inline void setNumPos(int i) {
		numInstancesPos = i;
	}

	inline void setNumOK(int i) {
		numInstancesPosOK = i;
	}
};

#endif
