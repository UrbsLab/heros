#ifndef _CLASSIFIER_HYPERRECT_SSE_
#define _CLASSIFIER_HYPERRECT_SSE_

#include "classifier.h"
#include "timerRealKR.h"
#include "timerGlobals.h"
#include "agentPerformanceTraining.h"

#include <emmintrin.h>
#include "macros_sse.h"

extern timerRealKR *tReal;
extern timerGlobals *tGlobals;

class classifier_hyperrect_sse: public classifier   {
	aligned_float *chromosome;

	/* Operadors de crossover i mutation */
	void crossover_1px(classifier_hyperrect_sse *in1,classifier_hyperrect_sse *in2
		,classifier_hyperrect_sse *out1,classifier_hyperrect_sse *out2);
	double getGene(short int attr,short int value); 
	void setGene(short int attr,short int value,double nfo);
	double mutationOffset(double geneValue,double offsetMin,double offsetMax);
	void initializeChromosome(void);

public:
	classifier_hyperrect_sse();
	classifier_hyperrect_sse(const classifier_hyperrect_sse &orig,int son=0);
	~classifier_hyperrect_sse();

	inline int getClass() {
        	return (int) chromosome[tReal->ruleSize - 1];
	}

	inline int doMatch(instance * ins)
	{
		int i,j;
		__m128i vecRes,vecTmp,vecOne;
		__m128 v1,v2,v3;
		vecOne=(__m128i){-1,-1};

		for(j=0;j<tReal->sizeBounds;j+=tReal->attPerBlock) {
			VEC_MATCH(v1,&chromosome[j],v2
				,&chromosome[j+tReal->sizeBounds],v3
				,&ins->realValues[j],vecTmp,vecOne,vecRes);
			if(!(0xFFFF==_mm_movemask_epi8(vecRes))) return 0;
		}

		return 1;
	}

	double computeTheoryLength();
	void crossover(classifier *,classifier *,classifier *);
	void mutation();
	void dumpPhenotype(char *string);

	inline int numSpecialStages(){return 0;}
	void doSpecialStage(int stage){}

	void postprocess();
};

#endif

