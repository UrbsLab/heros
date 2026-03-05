#ifndef _TIMER_EVOLUTION_STATS_H_
#define _TIMER_EVOLUTION_STATS_H_

#include "timingProcess.h"
#include "JVector.h"

class timerEvolutionStats: public timingProcess {
        int maxMin;
        int iterationsSinceBest;
        int globalIterationsSinceBest;
        double bestFitness;
	int doDumpStats;

        void bestOfIteration(int iteration,double bestFitness,double bestAcc);

public:
        int getIterationsSinceBest();
        int getGlobalIterationsSinceBest();
        void resetBestStats();

	timerEvolutionStats();
	~timerEvolutionStats();
	void initialize(populationWrapper *pPW) {pw=pPW;}
	void newIteration(int iteration,int finalIteration) {}
	void dumpStats(int iteration);
	void reinit();
};

#endif
