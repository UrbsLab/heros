#ifndef _CLASSIFIER_HYPERRECT_
#define _CLASSIFIER_HYPERRECT_

#include "classifier.h"
#include "timerRealKR.h"
#include "timerGlobals.h"
#include "agentPerformanceTraining.h"

extern timerRealKR *tReal;
extern timerGlobals *tGlobals;

class classifier_hyperrect: public classifier   {
	float *chromosome;

	/* Operadors de crossover i mutation */
	void crossover_1px(classifier_hyperrect *in1,classifier_hyperrect *in2
		,classifier_hyperrect *out1,classifier_hyperrect *out2);
	float getGene(short int attr,short int value); 
	void setGene(short int attr,short int value,float nfo);
	float mutationOffset(float geneValue,float offsetMin,float offsetMax);
	void initializeChromosome(void);

public:
	classifier_hyperrect();
	classifier_hyperrect(const classifier_hyperrect &orig,int son=0);
	~classifier_hyperrect();

	inline int getClass() {
        	return (int) chromosome[tReal->ruleSize - 1];
	}

	inline void swapD(float &a,float &b) {
	        float temp=a;
	        a=b;
	        b=temp;
	}


	inline int doMatch(instance * ins)
	{
		float *ptr = chromosome;
		int j, match;
		int numAtributs = tGlobals->numAttributesMC;

		float *cAtr = ptr;
		for (j = 0; j < numAtributs; j++) {
			if (ai.getTypeOfAttribute(j) == REAL) {
				float valueAtr = ins->realValues[j];
				float min = cAtr[0];
				float max = cAtr[1];
				if (min<max && ((valueAtr < min) || (valueAtr > max)))
					return 0;
			} else {
				int valueAtr = (unsigned char)ins->realValues[j];
				if (cAtr[valueAtr] == 0) {
					return 0;
				}
			}
			cAtr += tReal->attributeSize[j];
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

