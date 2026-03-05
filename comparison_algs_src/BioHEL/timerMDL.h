#ifndef _TIMER_MDL_H_
#define _TIMER_MDL_H_

#include "timingProcess.h"

class agentPerformanceTraining;
class classifier;

class timerMDL: public timingProcess {
	int startIteration;
	double mdlWeight;
	double mdlWeightRelaxFactor;
	double initialTheoryLenghtRatio;
	int fixedWeight;
	int mdlWeightRelaxStopIteration;
	double mdlWeightRelaxStopAccuracy;
	int iterationMDL;

public:
	int activated;
	int mdlAccuracy;
        double coverageBreak;
        double coverageRatio;
        double *coverageBreaks;

         int numAttsK;

	double mdlFitness(classifier &ind,agentPerformanceTraining *ap);
	timerMDL();
	void initialize(populationWrapper *pPW) {pw=pPW;}
	void newIteration(int iteration,int finalIteration);
	void dumpStats(int iteration);
        void adjustCovBreak(double cov);
        void adjustCovBreak(double cov, int clas);

//         void adjustNumAttsK(int k);
	void reinit();

        double getCoverageBreak() {
                return coverageBreak;
        }

};

#endif
