#include "classifier_gabil.h"
#include "random.h"
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include "attributesInfo.h"
#include "timerSymbolicKR.h"
#include "timerHierar.h"
#include "timerCrossover.h"
#include "timerMDL.h"
#include "timerGlobals.h"
#include "instanceSet.h"
#include "sampling.h"

extern attributesInfo ai;
extern timerSymbolicKR *tSymbolic;
extern timerHierar *tHierar;
extern timerGlobals *tGlobals;
extern timerCrossover *tCross;
extern timerMDL *tMDL;
extern Random rnd;
extern instanceSet *is;
extern int lastIteration;

using namespace std;

double classifier_gabil::computeTheoryLength()
{
	int j, k;
	unsigned char *ptr = chromosome;
	theoryLength = 0;

	for (j = 0; j < tGlobals->numAttributesMC; j++) {
		int numFalse = 0;
		int numValues = tSymbolic->sizeAttribute[j];
		for (k = 0; k < numValues; k++) {
			if (!ptr[k]) numFalse++;
		}
		theoryLength += (double)numFalse/(double)numValues; 
		ptr+=numValues;
	}
	theoryLength/=(double)tGlobals->numAttributesMC;

	return theoryLength;
}

classifier_gabil::classifier_gabil()
{
	int i;

	length = tSymbolic->ruleSize;
	chromosome = new unsigned char[length];

	classifier_gabil::initializeChromosome();
}

classifier_gabil::
classifier_gabil(const classifier_gabil & orig, int son)
{
	*this = orig;

	if (!son) {
		chromosome = new unsigned char[length];
		bcopy(orig.chromosome, chromosome,
		      length * sizeof(unsigned char));
	} else {
		chromosome = NULL;
	}
}

classifier_gabil::~classifier_gabil()
{
	delete chromosome;
}

void classifier_gabil::gabilRuleCover(unsigned char *rule,instance *ins,double prob)
{
	int j, k;

	unsigned char *ptr=rule;
	for (j = 0; j < tGlobals->numAttributesMC; j++) {
		int value;
		if(ins) value=(unsigned char)ins->realValues[j];
		else value = -1;
		for (k = 0; k < tSymbolic->sizeAttribute[j]; k++){
			if(k!=value) {
				if (!rnd < prob) ptr[k]=1;
				else ptr[k]=0;
			} else {
				ptr[k]=1;
			}
		}
		ptr+=tSymbolic->sizeAttribute[j];
	}

	if(ins) {
		ptr[0]=ins->getClass();
	} else {
		int cl;
		do {
			cl=rnd(0,ai.getNumClasses()-1);
		} while(tGlobals->defaultClassPolicy!=DISABLED 
			&& cl==tGlobals->defaultClass);
		ptr[0]=cl;
	}
}

void classifier_gabil::initializeChromosome()
{
	int i;

	unsigned char *ptr=chromosome;
	instance *ins=NULL;
	if(tGlobals->smartInit) {
		if(tGlobals->defaultClassPolicy!=DISABLED) {
			ins=is->getInstanceInit(tGlobals->defaultClass);
		} else {
			ins=is->getInstanceInit(ai.getNumClasses());
		}
	}

	gabilRuleCover(ptr,ins,tGlobals->probOne);
}

void classifier_gabil::setGene(short int gene, short int value
	, unsigned char nfo)
{
	chromosome[tSymbolic->offsetAttribute[gene] + value] = nfo;
	modif = 1;
}

unsigned char classifier_gabil::getGene(short int gene, short int value)
{
	return chromosome[tSymbolic->offsetAttribute[gene] + value];
}

void classifier_gabil::crossover(classifier * in,
				   classifier * out1, classifier * out2)
{

	switch (tCross->cxOperator) {
		case CROSS_INFORMED:
			crossover_informed(this, (classifier_gabil *) in,
			      (classifier_gabil *) out1,
			      (classifier_gabil *) out2);
			break;
		case CROSS_2P:
			crossover_2px(this, (classifier_gabil *) in,
			      (classifier_gabil *) out1,
			      (classifier_gabil *) out2);
			break;
		case CROSS_1P:
		default:
			crossover_1px(this, (classifier_gabil *) in,
			      (classifier_gabil *) out1,
			      (classifier_gabil *) out2);
	}

}

void classifier_gabil::mutation()
{
	int i;
	int attribute, value;

	// Modificarem el chromosome
	modif = 1;

	if(tGlobals->numClasses>1 && !rnd<0.10) {
		attribute = tGlobals->numAttributesMC;
		value=0;
	} else {
		attribute=rnd(0,tGlobals->numAttributesMC-1);
		value=rnd(0,tSymbolic->sizeAttribute[attribute]-1);
	}

	if (attribute != tGlobals->numAttributesMC ) {
		if (getGene(attribute, value) == 0)
			setGene(attribute, value, 1);
		else
			setGene(attribute, value, 0);
	} else {
		int oldValue=getGene(attribute,value);
		int newValue;
		do {
			newValue = rnd(0, ai.getNumClasses()-1);
		} while(newValue==oldValue || tGlobals->defaultClassPolicy!=DISABLED && newValue==tGlobals->defaultClass);
		setGene(attribute, value, newValue);
	}
}

static void swap(int &a, int &b)
{
	int tmp;

	tmp = a;
	a = b;
	b = tmp;
}

void classifier_gabil::crossover_2px(classifier_gabil * in1,
				       classifier_gabil * in2,
				       classifier_gabil * out1,
				       classifier_gabil * out2)
{
	out1->modif = out2->modif = 1;
	out1->length = tSymbolic->ruleSize;
	out2->length = tSymbolic->ruleSize;
	out1->chromosome = new unsigned char[tSymbolic->ruleSize];
	out2->chromosome = new unsigned char[tSymbolic->ruleSize];

	int cutPoint1 = rnd(0, tSymbolic->ruleSize - 1);
	int cutPoint2 = rnd(0, tSymbolic->ruleSize - 1);
	if (cutPoint1 > cutPoint2) swap(cutPoint1, cutPoint2);

	bcopy(in1->chromosome, out1->chromosome, cutPoint1);
	bcopy(in2->chromosome, out2->chromosome, cutPoint1);

	bcopy(&in1->chromosome[cutPoint1], &out2->chromosome[cutPoint1], cutPoint2-cutPoint1);
	bcopy(&in2->chromosome[cutPoint1], &out1->chromosome[cutPoint1], cutPoint2-cutPoint1);

	bcopy(&in1->chromosome[cutPoint2], &out1->chromosome[cutPoint2], tSymbolic->ruleSize-cutPoint2);
	bcopy(&in2->chromosome[cutPoint2], &out2->chromosome[cutPoint2], tSymbolic->ruleSize-cutPoint2);
}

void classifier_gabil::dumpPhenotype(char *string)
{
	unsigned char *ptr = chromosome;
	char temp[10000];
	char temp2[1000];
	int i, j, k;

	strcpy(string, "");
	unsigned char *ptr2 = ptr;
	for (j = 0; j < tGlobals->numAttributesMC; j++) {
		sprintf(temp,"Att %s is "
			,ai.getAttributeName(j)->cstr());
		int irr=1;
		for (k = 0; k < tSymbolic->sizeAttribute[j]; k++) {
			if(ptr2[k]) {
				sprintf(temp2,"%s,",ai.getNominalValue(j,k)->cstr());
				strcat(temp,temp2);
			} else {
				irr=0;
			}
		}

		if(!irr) {
			if(temp[strlen(temp) - 1] == ',')
				temp[strlen(temp) - 1] = 0;
			strcat(string, temp);
			strcat(string, "|");
		}
		ptr2 += tSymbolic->sizeAttribute[j];
	}
	int cl=(int)*ptr2;
	sprintf(temp, "%s\n",
		ai.getNominalValue(tGlobals->numAttributesMC
			, cl)->cstr());
	strcat(string, temp);
}

void classifier_gabil::crossover_informed(classifier_gabil * in1,
				       classifier_gabil * in2,
				       classifier_gabil * out1,
				       classifier_gabil * out2) {
	out1->modif = out2->modif = 1;
	out1->length = tSymbolic->ruleSize;
	out2->length = tSymbolic->ruleSize;
	out1->chromosome = new unsigned char[tSymbolic->ruleSize];
	out2->chromosome = new unsigned char[tSymbolic->ruleSize];

	int i,j;
	
	for(i=0;i<tCross->numBB;i++) {
		// Let's choose parent...
		if(!rnd<0.5) {
			for(j=0;j<tCross->sizeBBs[i];j++) {
				int att=tCross->defBBs[i][j];
				bcopy(&in1->chromosome[tSymbolic->offsetAttribute[att]]
					,&out1->chromosome[tSymbolic->offsetAttribute[att]]
					,sizeof(unsigned char)
						*tSymbolic->sizeAttribute[att]);
				bcopy(&in2->chromosome[tSymbolic->offsetAttribute[att]]
					,&out2->chromosome[tSymbolic->offsetAttribute[att]]
					,sizeof(unsigned char)
						*tSymbolic->sizeAttribute[att]);
			}
		} else {
			for(j=0;j<tCross->sizeBBs[i];j++) {
				int att=tCross->defBBs[i][j];
				bcopy(&in1->chromosome[tSymbolic->offsetAttribute[att]]
					,&out2->chromosome[tSymbolic->offsetAttribute[att]]
					,sizeof(unsigned char)
						*tSymbolic->sizeAttribute[att]);
				bcopy(&in2->chromosome[tSymbolic->offsetAttribute[att]]
					,&out1->chromosome[tSymbolic->offsetAttribute[att]]
					,sizeof(unsigned char)
						*tSymbolic->sizeAttribute[att]);
			}
		}
	}
}

void classifier_gabil::crossover_1px(classifier_gabil * in1,
				       classifier_gabil * in2,
				       classifier_gabil * out1,
				       classifier_gabil * out2)
{
	int cutPoint;

	out1->modif = out2->modif = 1;

	cutPoint = rnd(0, tSymbolic->ruleSize - 1);
	out1->length = tSymbolic->ruleSize;
	out2->length = tSymbolic->ruleSize;

	out1->chromosome = new unsigned char[tSymbolic->ruleSize];
	out2->chromosome = new unsigned char[tSymbolic->ruleSize];

	bcopy(in1->chromosome, out1->chromosome, cutPoint);
	bcopy(in2->chromosome, out2->chromosome, cutPoint);

	bcopy(&in1->chromosome[cutPoint], &out2->chromosome[cutPoint], tSymbolic->ruleSize-cutPoint);
	bcopy(&in2->chromosome[cutPoint], &out1->chromosome[cutPoint], tSymbolic->ruleSize-cutPoint);
}

void classifier_gabil::postprocess()
{
	int numInstances = is->getNumInstancesOfIteration();	
	int instanceMatched[numInstances];

	cleanRule(instanceMatched);
	generalizeRule(instanceMatched);
}

void classifier_gabil::generalizeRule(int *instanceMap)
{
	int i,j;
	int len=tSymbolic->ruleSize-1;
	int countPos[len];
	int countNeg[len];
	JVector<int> *candInst[len];
	int numInstances=is->getNumInstancesOfIteration();
	instance **instances=is->getInstancesOfIteration();
	int exitLoop=0;
	int lastRound=0;

	do {
		if(lastRound) {
			if(lastRound==3) exitLoop=1;
			lastRound++;
		}

		for(i=0;i<len;i++) {
			countPos[i]=countNeg[i]=0;
			candInst[i]=NULL;
		}
	
		int cl=chromosome[len];
		for(i=0;i<numInstances;i++) {
			if(!instanceMap[i]) {
				int numMatches=0;
				int whichPos;
				for(j=0;j<tGlobals->numAttributesMC;j++) {
					int value=(unsigned char)instances[i]->realValues[j];
					int pos=tSymbolic->offsetAttribute[j]+value;
					if(chromosome[pos]==0) {
						numMatches++;
						whichPos=pos;
						if(numMatches>=2) break;
					}
				}
		
				if(numMatches==1) {
					if(candInst[whichPos]==NULL) {
						candInst[whichPos] = new JVector<int>((int)(numInstances*0.1));
					}
					candInst[whichPos]->addElement(i);
					if(instances[i]->instanceClass==cl) {
						countPos[whichPos]++;
					} else {
						countNeg[whichPos]++;
					}
				}
			}
		}
		
		int max=0;
		int bestPos;
		if(lastRound) {
			for(i=0;i<len;i++) {
				int diff=countPos[i]-countNeg[i];
				if(diff>max) {
					bestPos=i;
					max=diff;
				}
			}
		} else {
			for(i=0;i<len;i++) {
				if(countNeg[i]==0 && countPos[i]>max) {
					bestPos=i;
					max=countPos[i];
				}
			}
		}
	
		if(max>0) {
			printf("Best %d %d\n",countPos[bestPos],countNeg[bestPos]);
			chromosome[bestPos]=1;
			for(i=0;i<max;i++) {
				instanceMap[candInst[bestPos]->elementAt(i)]=1;
			}
		} else {
			if(lastRound) exitLoop=1;
			else lastRound=1;
			//exitLoop=1;
		}
	
		for(i=0;i<len;i++) {
			if(candInst[i]) delete candInst[i];
		}
	} while(!exitLoop);
}

void classifier_gabil::cleanRule(int *instanceMatched)
{
	int numInstances = is->getNumInstancesOfIteration();	
	instance **instances=is->getInstancesOfIteration();
	int i,j,k;

	for(i=0;i<numInstances;i++) {
		instanceMatched[i]=1;
	}

	unsigned char *ptr=chromosome;	
	int cl=ptr[tSymbolic->ruleSize - 1];
	int size=tSymbolic->ruleSize - 1;
	int numPossibleCandidates;
	int exitLoop=0;
	int lastRound=0;

	do {
		if(lastRound) {
			if(lastRound==3) exitLoop=1;
			lastRound++;
		}

		int posMatch[size];
		int negMatch[size];
		int candidates[size];
	
		for(j=0;j<size;j++) {
			posMatch[j]=negMatch[j]=candidates[j]=0;
		}
		
		for(j=0;j<numInstances;j++) {
			if(instanceMatched[j]) {
				if(doMatch(instances[j])) {
					if(instances[j]->instanceClass==cl) {
						for(k=0;k<tGlobals->numAttributesMC;k++) {
							int pos=tSymbolic->offsetAttribute[k]+(unsigned char)instances[j]->realValues[k];
							posMatch[pos]++;
							candidates[pos]=0;
						}
					} else {
						for(k=0;k<tGlobals->numAttributesMC;k++) {
							int pos=tSymbolic->offsetAttribute[k]+(unsigned char)instances[j]->realValues[k];
							negMatch[pos]++;
							if(!posMatch[pos]) {
								candidates[pos]=1;
							}
						}
					}
				} else {
					instanceMatched[j]=0;
				}
			}
		}
	
		if(lastRound) {
			int posMax;
			double maxNeg=1;

			for(j=0;j<size;j++) {
				double acc=(double)posMatch[j]/(double)(posMatch[j]+negMatch[j]);
				if(acc<maxNeg) {
					posMax=j;
					maxNeg=acc;
				}
			}
			if(maxNeg<1) {
				ptr[posMax]=0;
			}
		} else {
			int maxNeg=-1;
			int posMax;

			for(j=0;j<size;j++) {
				if(candidates[j]) {
					if(negMatch[j]>maxNeg) {
						posMax=j;
						maxNeg=negMatch[j];
					}
				}
			}

			if(maxNeg>-1) {
				ptr[posMax]=0;
			} else {
				if(lastRound) exitLoop=1;
				else lastRound=1;
				//exitLoop=1;
			}
		}

	} while(!exitLoop);	
}
