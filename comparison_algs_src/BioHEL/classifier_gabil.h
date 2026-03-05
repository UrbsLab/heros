#ifndef _CLASSIFIER_GABIL_

#define _CLASSIFIER_GABIL_

#include "classifier.h"
#include "agentPerformanceTraining.h"
#include "timerGlobals.h"
#include "timerSymbolicKR.h"


extern timerGlobals *tGlobals;
extern timerSymbolicKR *tSymbolic;

class classifier_gabil: public classifier   {

	void crossover_informed(classifier_gabil *in1
		,classifier_gabil *in2
		,classifier_gabil *out1
		,classifier_gabil *out2);
	void crossover_1px(classifier_gabil *in1
		,classifier_gabil *in2
		,classifier_gabil *out1
		,classifier_gabil *out2);
	void crossover_2px(classifier_gabil *in1,classifier_gabil *in2
		, classifier_gabil *out1
		,classifier_gabil *out2);
	unsigned char getGene(short int gene,short int value); 
	void setGene(short int gene,short int value
		,unsigned char nfo);
	void gabilRuleCover(unsigned char *rule,instance *ins,double prob);

	void initializeChromosome(void);

	void cleanRule(int *instanceMatched);
	void generalizeRule(int *instanceMatched);


	inline virtual void dumpGenotype(char *string) 
	{
		unsigned char *ptr=chromosome;
		int i,j,k;

		string[0]=0;
		char tmp[tSymbolic->ruleSize+tGlobals->numAttributes];
		int countPos=0;
		int dead=0;
		for (j=0;j<tGlobals->numAttributesMC && !dead;j++) {
			int attDead=1;
			for(k=tSymbolic->offsetAttribute[j];k<tSymbolic->offsetAttribute[j]+tSymbolic->sizeAttribute[j];k++) {
				if(ptr[k]==1) attDead=0;
				tmp[countPos++]=ptr[k]+'0';
			}
			tmp[countPos++]='|';
			if(attDead==1) dead=1;
		}
	
		if(!dead) {
			tmp[countPos++]=ptr[tSymbolic->ruleSize-1]+'0';
			tmp[countPos]=0;
			strcat(string,tmp);
			strcat(string,"\n");
		}
	}
public:
  	unsigned char *chromosome;

	classifier_gabil();
	classifier_gabil(const classifier_gabil &orig,int son=0);
	~classifier_gabil();

	inline int getClass() {
		return (int) chromosome[tSymbolic->ruleSize - 1];
	}

	inline int doMatch(instance *ins) 
	{
		int j;

		for (j = 0; j < tGlobals->numAttributesMC; j++) {
			if (chromosome[tSymbolic->offsetAttribute[j]
				   //+ ins->nominalValues[j]] == 0) {
				   + (unsigned char)ins->realValues[j]] == 0) {
				return 0;
			}
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

