#ifndef _CLASSIFIER_HYPERRECT_LIST_
#define _CLASSIFIER_HYPERRECT_LIST_

#include "classifier.h"
#include "timerRealKR.h"
#include "timerGlobals.h"
#include "agentPerformanceTraining.h"

extern attributesInfo ai;
extern timerRealKR *tReal;
extern timerGlobals *tGlobals;

class classifier_hyperrect_list: public classifier   {

	void crossover_1px(classifier_hyperrect_list *in1,classifier_hyperrect_list *in2
		,classifier_hyperrect_list *out1,classifier_hyperrect_list *out2);
	float mutationOffset(float geneValue,float offsetMin,float offsetMax);
	void initializeChromosome(int empty);

public:
	float *predicates;
	int *offsetPredicates;
	int numAtt;
	int *whichAtt;
	int classValue;
	int ruleSize;
	int numDiscrete;
	int *listDiscretePos;
	int *listDiscreteAtt;
	int numReal;
	int *listRealPos;
	int *listRealAtt;

	classifier_hyperrect_list(int empty=0);
	classifier_hyperrect_list(const classifier_hyperrect_list &orig,int son=0);
	~classifier_hyperrect_list();

        inline void swapD(float &a,float &b) {
                float temp=a;
                a=b;
                b=temp;
        }


	inline int getClass() {
        	return classValue;
	}

	inline int doMatch(instance * ins)
	{
		int i;

		for(i=0;i<numReal;i++) {
			int base=offsetPredicates[listRealPos[i]];
			register float value=ins->realValues[listRealAtt[i]];
			if(value<predicates[base] || value>predicates[base+1]) return 0;
		}

		for(i=0;i<numDiscrete;i++) {
			int base=offsetPredicates[listDiscretePos[i]];
			register int value=(unsigned char)ins->realValues[listDiscreteAtt[i]];
			if(predicates[base+value]==0) return 0;
		}

		return 1;
	}

	double computeTheoryLength();
	void crossover(classifier *,classifier *,classifier *);
	void mutation();
	void dumpPhenotype(char *string);

	inline int numSpecialStages(){return 2;}
	void doSpecialStage(int);

	void postprocess();
	
	void initiateEval();
	void finalizeEval();


};

#endif

