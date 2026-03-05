#ifndef _TIMER_REAL_KR_H_
#define _TIMER_REAL_KR_H_

#include "timingProcess.h"

typedef struct {
	int count;
	int pos;
} rankAtt;

class evaluator {
	public:
	virtual int evaluate(float *,float)=0;
};

class evaluatorReal : public evaluator {
	public:
	inline int evaluate(float *term,float value) {
        	return (value<term[0] || value>term[1]);
	}
};

class evaluatorNominal : public evaluator {
	public:
	inline int evaluate(float *term,float value) {
		return (term[(unsigned char)value]==0);
	}
};

class timerRealKR: public timingProcess {
	int enabled;
public:
	int *attributeSize;
	int *attributeOffset;
	int ruleSize;
	float dOfFR;
        float alphaOfBLX;
        float nOfSBX;
	int thereIsSpecialCrossover;
	double instanceTheoryLength;
	double probSharp;
	double probIrr;
	double coverageInit;

	evaluator **evaluators;

	int hyperrectList;
	int *fitGen;
	int *fitSpe;

	double probGeneralizeList;
	double probSpecializeList;
	
	int attPerBlock;
	int sizeBounds;
	int rotateIntervals;
	float *sinTable;
	float *cosTable;
	float stepRatio;
	float *minD,*maxD;
	float *sizeD;
	int numSteps;
	int mutSteps;
	int step0;
	int numAngles;
	int *angleList1;
	int *angleList2;
	double prob0AngleInit;
	double prob0AngleMut;
	int numUsedAngles;
	int sizeAngles;

	timerRealKR();
	void initialize(populationWrapper *pPW) {pw=pPW;}
	void newIteration(int iteration,int finalIteration);
	void dumpStats(int iteration);

	void crossoverBLX(float parent1,float parent2,float &son1
		,float &son2,float minDomain,float maxDomain);
	void crossoverSBX(float parent1,float parent2,float &son1,float &son2
		,float minDomain,float maxDomain);
	void crossoverFR(float parent1,float parent2,float &son1,float &son2
		,float minDomain,float maxDomain);
	float crossoverFRp(float minPare,float maxPare,float interval
		,float minDomain,float maxDomain);

	void specialCrossover(float parent1,float parent2,float &son1
		,float &son2,float minDomain,float maxDomain);
	float fmin(float a,float b);
	float fmax(float a,float b);
};

#endif
