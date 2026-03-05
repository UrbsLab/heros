#ifndef _CLASSIFIER_ROTATED_HYPERRECT_
#define _CLASSIFIER_ROTATED_HYPERRECT_

#include "classifier.h"
#include "timerRealKR.h"
#include "timerGlobals.h"
#include "agentPerformanceTraining.h"

#include <emmintrin.h>
#include "macros_sse.h"

extern timerRealKR *tReal;
extern timerGlobals *tGlobals;

class classifier_rotated_hyperrect: public classifier   {
	float *chromosome;

	/* Operadors de crossover i mutation */
	void crossover_1px(classifier_rotated_hyperrect *in1,classifier_rotated_hyperrect *in2
		,classifier_rotated_hyperrect *out1,classifier_rotated_hyperrect *out2);
	double getGene(short int attr,short int value); 
	void setGene(short int attr,short int value,double nfo);
	double mutationOffset(double geneValue,double offsetMin,double offsetMax);
	void initializeChromosome(void);

public:
	classifier_rotated_hyperrect();
	classifier_rotated_hyperrect(const classifier_rotated_hyperrect &orig,int son=0);
	~classifier_rotated_hyperrect();

	inline int getClass() {
        	return (int) chromosome[tReal->ruleSize - 1];
	}

	inline int doMatch(instance * ins)
	{
		int i,j;
                __m128i vecRes,vecTmp,vecOne;
		__m128 v1,v2,v3;
		vecOne=(__m128i){-1,-1};
		float realValues[tReal->sizeBounds];

		bcopy(ins->realValues,realValues,sizeof(float)*tReal->sizeBounds);

                int index=tReal->sizeBounds*2;
                for(i=0;i<tReal->numUsedAngles;i++,index+=2) {
			register int angle=(int)chromosome[index+1];
                        if(angle!=tReal->step0) {
				int pos1=tReal->angleList1[(int)chromosome[index]];
				int pos2=tReal->angleList2[(int)chromosome[index]];
				float newX=realValues[pos1]*tReal->cosTable[angle]
					-realValues[pos2]*tReal->sinTable[angle];
				float newY=realValues[pos1]*tReal->sinTable[angle]
					+realValues[pos2]*tReal->cosTable[angle];
				realValues[pos1]=newX;
				realValues[pos2]=newY;
                	}
               	}

		/*for(i=0;i<tGlobals->numAttributesMC;i++) {
			if(realValues[i]<tReal->minD[i] || realValues[i]>tReal->maxD[i]) {
				char string[10000];
				dumpPhenotype(string);
				printf("Rule %s produces out of range value %d:%f for instance\n",string,i,realValues[i]);
				ins->dumpInstance();
				for(j=0;j<tGlobals->numAttributesMC;j++) {
					printf("%.3f ",realValues[j]);
				}
				printf("\n");
				exit(1);
			}
		}*/

                for(j=0;j<tReal->sizeBounds;j+=tReal->attPerBlock) {
                        VEC_MATCH(v1,&chromosome[j],v2
                                ,&chromosome[j+tReal->sizeBounds],v3
                                ,&realValues[j],vecTmp,vecOne,vecRes);
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

