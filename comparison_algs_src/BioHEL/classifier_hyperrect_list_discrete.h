#ifndef _CLASSIFIER_HYPERRECT_LIST_DISCRETE_
#define _CLASSIFIER_HYPERRECT_LIST_DISCRETE_

#include "classifier.h"
#include "timerRealKR.h"
#include "timerGlobals.h"
#include "agentPerformanceTraining.h"

extern attributesInfo ai;
extern timerRealKR *tReal;
extern timerGlobals *tGlobals;

class classifier_hyperrect_list_discrete: public classifier   {

	void crossover_1px(classifier_hyperrect_list_discrete *in1,classifier_hyperrect_list_discrete *in2
		,classifier_hyperrect_list_discrete *out1,classifier_hyperrect_list_discrete *out2);
	float mutationOffset(float geneValue,float offsetMin,float offsetMax);
	void initializeChromosome(int empty=0);
         void initializeChromosomeNumRep(int numRep);

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


        classifier_hyperrect_list_discrete(int numRep=-1);
	classifier_hyperrect_list_discrete(const classifier_hyperrect_list_discrete &orig,int son=0);
	~classifier_hyperrect_list_discrete();

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

		for(i=0;i<numDiscrete;i++) {
			int base=offsetPredicates[listDiscretePos[i]];
			register int value=(unsigned char)ins->realValues[listDiscreteAtt[i]];
			if(predicates[base+value]==0) return 0;
		}

		/*for(i=0;i<numAtt;i++) {
			int base=offsetPredicates[i];
			int att=whichAtt[i];
			if(tReal->evaluators[att]->evaluate(&predicates[base],ins->realValues[att])) return 0;*/
			/*if (ai.getTypeOfAttribute(att) == REAL) {
				register float value=ins->realValues[att];
				if(value<predicates[base] || value>predicates[base+1]) return 0;
			} else {
				register int value=(unsigned char)ins->realValues[att];
				if(predicates[base+value]==0) return 0;
			}*/
		//}
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

	int equals(classifier *ind2);

        inline int getNumAtts() {
            printf("this num att\n");
            return numAtt;
        }

};

#endif

