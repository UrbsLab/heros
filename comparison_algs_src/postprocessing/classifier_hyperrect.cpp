#include "classifier_hyperrect.h"
#include "random.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <strings.h>
#include "attributesInfo.h"
#include "timerGlobals.h"
#include "timerMutation.h"
#include "timerCrossover.h"
#include "timerMDL.h"
#include "timerHierar.h"
#include "timerRealKR.h"
#include "instanceSet.h"

extern attributesInfo ai;
extern timerGlobals *tGlobals;
extern timerHierar *tHierar;
extern timerMutation *tMut;
extern timerCrossover *tCross;
extern timerMDL *tMDL;
extern timerRealKR *tReal;
extern instanceSet *is;
extern Random rnd;

double classifier_hyperrect::computeTheoryLength()
{
	int i, j, k;
	float *ptr = chromosome;
	theoryLength = 0.0;

	for (j = 0; j < tGlobals->numAttributesMC; j++) {
		if (ai.getTypeOfAttribute(j) == REAL) {
			float vmin=ptr[0];
			float vmax=ptr[1];
			float size=ai.getSizeDomain(j);
			if(vmin<vmax && size>0) {
				theoryLength += 1.0 - 
					(vmax-vmin) / size;
			}
		} else {
			double countFalses = 0;
			int numValues=tReal->attributeSize[j];
			for(k=0;k<numValues;k++) {
				if(!ptr[k]) countFalses++;
			}
			theoryLength+=(double)countFalses/(double)numValues;
		}
		ptr += tReal->attributeSize[j];
	}

	theoryLength/=(double)tGlobals->numAttributesMC;

	return theoryLength;
}

classifier_hyperrect::classifier_hyperrect()
{
	length = tReal->ruleSize;
	chromosome = new float[tReal->ruleSize];

	classifier_hyperrect::initializeChromosome();
}

classifier_hyperrect::classifier_hyperrect(const classifier_hyperrect & orig, int son)
{
	*this = orig;

	if (!son) {
		chromosome = new float[tReal->ruleSize];
		bcopy(orig.chromosome, chromosome,
		      tReal->ruleSize * sizeof(float));
	} else {
		chromosome = NULL;
	}
}

classifier_hyperrect::~classifier_hyperrect()
{
	delete chromosome;
}

void classifier_hyperrect::initializeChromosome()
{
	int i, j, k;

	instance *ins=NULL;
	if(tGlobals->smartInit) {
		if(tGlobals->defaultClassPolicy!=DISABLED) {
			ins=is->getInstanceInit(tGlobals->defaultClass);
		} else {
			ins=is->getInstanceInit(ai.getNumClasses());
		}
	}

	for (j = 0; j < tGlobals->numAttributesMC; j++) {
		if (ai.getTypeOfAttribute(j) == REAL) {
			float min,max;
			float sizeD=ai.getSizeDomain(j);
			float minD=ai.getMinDomain(j);
			float maxD=ai.getMaxDomain(j);
			if(!rnd<tReal->probIrr) {
				max=!rnd*sizeD+minD;
				min=!rnd*(maxD-max)+max;
			} else {
				//float size=(!rnd*0.10+0.90)*ai.getSizeDomain(j);
				float size=(!rnd*0.5+0.25)*sizeD;
				if(ins) {
					float val=ins->realValueOfAttribute(j);
					min=val-size/2.0;
					max=val+size/2.0;
					if(min<minD) {
						max+=(minD-min);
						min=minD;
					}
					if(max>maxD) {
						min-=(max-maxD);
						max=maxD;
					}
				} else {
					min=!rnd*(sizeD-size)+minD;
					max=min+size;
				}
			}
			
			setGene(j,0,min);
			setGene(j,1,max);
		} else {
			int value;
			if(ins) value=ins->valueOfAttribute(j);
			else value=-1;
			for(k=0;k<tReal->attributeSize[j];k++) {
				if(k!=value) {
					if(!rnd<tGlobals->probOne) {
						setGene(j,k,1);
					} else {
						setGene(j,k,0);
					}
				} else {
					setGene(j,k,1);
				}
			}
		}
	}

	int cl;
	if(ins) {
		cl=ins->getClass();
	} else {
		do {
			cl=rnd(0,ai.getNumClasses()-1);
		} while(tGlobals->defaultClassPolicy!=DISABLED && cl==tGlobals->defaultClass);
	}
	setGene(tGlobals->numAttributesMC, 0, cl);
}

void classifier_hyperrect::setGene(short int gene, short int value, float nfo)
{
	chromosome[tReal->attributeOffset[gene] + value] = nfo;
	modif = 1;
}

float classifier_hyperrect::getGene(short int gene, short int value)
{
	return chromosome[tReal->attributeOffset[gene] + value];
}

void classifier_hyperrect::crossover(classifier * in,
				classifier * out1, classifier * out2)
{
	crossover_1px(this, (classifier_hyperrect *) in,
		      (classifier_hyperrect *) out1,
		      (classifier_hyperrect *) out2);
}

float classifier_hyperrect::mutationOffset(float geneValue, float offsetMin,
				       float offsetMax)
{
	float newValue;
	if (!rnd < 0.5) {
		newValue = geneValue + !rnd * offsetMax;
	} else {
		newValue = geneValue - !rnd * offsetMin;
	}
	return newValue;
}

void classifier_hyperrect::mutation()
{
	int i;
	int attribute, value;

	modif = 1;

	if(tGlobals->numClasses>1 && !rnd<0.10) {
		attribute = tGlobals->numAttributesMC;
		value=0;
	} else {
		attribute=rnd(0,tGlobals->numAttributesMC-1);
		value=rnd(0,tReal->attributeSize[attribute]-1);
	}

	if (attribute != tGlobals->numAttributesMC) {
		float oldValue=getGene(attribute,value);
		if (ai.getTypeOfAttribute(attribute) == REAL) {
			float newValue;
			float minOffset, maxOffset;
			minOffset = maxOffset =
			    0.5 * ai.getSizeDomain(attribute);
			newValue = mutationOffset(oldValue, minOffset,
						  maxOffset);
			if (newValue < ai.getMinDomain(attribute))
				newValue = ai.getMinDomain(attribute);
			if (newValue > ai.getMaxDomain(attribute))
				newValue = ai.getMaxDomain(attribute);
			setGene(attribute, value, newValue);
		} else {
			if(oldValue==1) setGene(attribute,value,0);
			else setGene(attribute,value,1);
		}
	} else {
		int newValue;
		int oldValue = (int) getGene(attribute, value);
		do {
			newValue = rnd(0, ai.getNumClasses()-1);
		} while (newValue == oldValue || tGlobals->defaultClassPolicy!=DISABLED && newValue==tGlobals->defaultClass);
		setGene(attribute, value, newValue);
	}
}

void classifier_hyperrect::dumpPhenotype(char *string)
{
	float *ptr = chromosome;
	char temp[10000];
	char temp2[10000];
	char tmp1[20];
	char tmp2[20];
	int numIntervals=0;
	int i, j, k;

	strcpy(string, "");
	float *ptr2 = ptr;
	for (j = 0; j < tGlobals->numAttributesMC; j++) {
                sprintf(temp,"Att %s is "
                        ,ai.getAttributeName(j)->cstr());
                int irr=1;

		if (ai.getTypeOfAttribute(j) == REAL) {
			float min = ptr2[0];
			float max = ptr2[1];
			float size = ai.getSizeDomain(j);

			float sizeD=max-min;
			if(min<max && sizeD<size) {
				irr=0;
				sprintf(tmp1, "%f", min);
				sprintf(tmp2, "%f", max);
				sprintf(temp2, "[%s,%s]", tmp1, tmp2);
				strcat(temp,temp2);
			}
		} else {
			for(k=0;k<tReal->attributeSize[j];k++) {
				if(ptr2[k]) {
					sprintf(temp2,"%s,",ai.getNominalValue(j,k)->cstr());
					strcat(temp,temp2);
				} else {
					irr=0;
				}
			}
			if(temp[strlen(temp) - 1] == ',') temp[strlen(temp) - 1] = 0;
		}
		if(!irr) {
			strcat(string, temp);
			strcat(string, "|");
		}
		ptr2 += tReal->attributeSize[j];
	}
	int cl=(int) (*ptr2);
	sprintf(temp, "%s\n", ai.getNominalValue(tGlobals->numAttributesMC,cl)->cstr());
	strcat(string, temp);
}

void classifier_hyperrect::crossover_1px(classifier_hyperrect * in1,
				    classifier_hyperrect * in2,
				    classifier_hyperrect * out1,
				    classifier_hyperrect * out2)
{
	out1->modif = out2->modif = 1;

	int cutPoint = rnd(0, tReal->ruleSize - 1);
	out1->length = tReal->ruleSize;
	out2->length = tReal->ruleSize;
	out1->chromosome = new float[tReal->ruleSize];
	out2->chromosome = new float[tReal->ruleSize];

	bcopy(in1->chromosome, out1->chromosome, cutPoint * sizeof(float));
	bcopy(in2->chromosome, out2->chromosome, cutPoint * sizeof(float));

	bcopy(&in1->chromosome[cutPoint], &out2->chromosome[cutPoint],
	      (tReal->ruleSize - cutPoint) * sizeof(float));
	bcopy(&in2->chromosome[cutPoint], &out1->chromosome[cutPoint],
	      (tReal->ruleSize - cutPoint) * sizeof(float));
}

void classifier_hyperrect::postprocess()
{
}

