#include "classifier_hyperrect_sse.h"
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

double classifier_hyperrect_sse::computeTheoryLength()
{
	int i, j, k;
	float *ptr = chromosome;
	theoryLength = 0.0;

	for (j = 0; j < tGlobals->numAttributesMC; j++) {
		float vmin=ptr[j];
		float vmax=ptr[tReal->sizeBounds+j];
		double size=ai.getSizeDomain(j);
		if(vmin<vmax && size>0) {
			theoryLength += 1.0 - (vmax-vmin) / size;
		}
	}

	theoryLength/=(double)tGlobals->numAttributesMC;

	return theoryLength;
}

classifier_hyperrect_sse::classifier_hyperrect_sse()
{
	length = tReal->ruleSize;
	chromosome = new aligned_float[tReal->ruleSize];
	classifier_hyperrect_sse::initializeChromosome();
}

classifier_hyperrect_sse::classifier_hyperrect_sse(const classifier_hyperrect_sse & orig, int son)
{
	*this = orig;

	if (!son) {
		chromosome = new aligned_float[tReal->ruleSize];
		bcopy(orig.chromosome, chromosome,
		      tReal->ruleSize * sizeof(float));
	} else {
		chromosome = NULL;
	}
}

classifier_hyperrect_sse::~classifier_hyperrect_sse()
{
	delete chromosome;
}

void classifier_hyperrect_sse::initializeChromosome()
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
		double min,max;
		double sizeD=ai.getSizeDomain(j);
		double minD=ai.getMinDomain(j);
		double maxD=ai.getMaxDomain(j);
		if(!rnd<tReal->probIrr) {
			max=!rnd*sizeD+minD;
			min=!rnd*(maxD-max)+max;
		} else {
			//double size=(!rnd*0.10+0.90)*ai.getSizeDomain(j);
			double size=(!rnd*0.5+0.25)*sizeD;
			if(ins) {
				double val=ins->realValueOfAttribute(j);
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
	}
	for(;j<tReal->sizeBounds;j++) {
		chromosome[j]=1;
		chromosome[j+tReal->sizeBounds]=0;
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

void classifier_hyperrect_sse::setGene(short int gene, short int value, double nfo)
{
	if(gene<tGlobals->numAttributesMC) {
		chromosome[gene + value*tReal->sizeBounds] = (float)nfo;
	} else {
		chromosome[tReal->ruleSize-1] = (float)nfo;
	}
	modif = 1;
}

double classifier_hyperrect_sse::getGene(short int gene, short int value)
{
	if(gene<tGlobals->numAttributesMC) {
		return chromosome[gene + value*tReal->sizeBounds];
	} else {
		return chromosome[tReal->ruleSize-1];
	}
}

void classifier_hyperrect_sse::crossover(classifier * in,
				classifier * out1, classifier * out2)
{
	crossover_1px(this, (classifier_hyperrect_sse *) in,
		      (classifier_hyperrect_sse *) out1,
		      (classifier_hyperrect_sse *) out2);
}

double classifier_hyperrect_sse::mutationOffset(double geneValue, double offsetMin,
				       double offsetMax)
{
	double newValue;
	if (!rnd < 0.5) {
		newValue = geneValue + !rnd * offsetMax;
	} else {
		newValue = geneValue - !rnd * offsetMin;
	}
	return newValue;
}

void classifier_hyperrect_sse::mutation()
{
	int i;
	int attribute=-1, value;

	modif = 1;

	if(tGlobals->numClasses>1 && !rnd<0.10) {
		attribute = tGlobals->numAttributesMC;
		value=0;
	} else {
		attribute=rnd(0,tGlobals->numAttributesMC-1);
		value=rnd(0,1);
	}

	if (attribute != tGlobals->numAttributesMC) {
		double oldValue=getGene(attribute,value);
		double newValue;
		double minOffset, maxOffset;
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
		int newValue;
		int oldValue = (int) getGene(attribute, value);
		do {
			newValue = rnd(0, ai.getNumClasses()-1);
		} while (newValue == oldValue || tGlobals->defaultClassPolicy!=DISABLED && newValue==tGlobals->defaultClass);
		setGene(attribute, value, newValue);
	}
}

void classifier_hyperrect_sse::dumpPhenotype(char *string)
{
	float *ptr = chromosome;
	char temp[10000];
	char temp2[10000];
	char tmp1[20];
	char tmp2[20];
	int numIntervals=0;
	int i, j, k;

	strcpy(string, "");
	for (j = 0; j < tGlobals->numAttributesMC; j++) {
                sprintf(temp,"Att %s is "
                        ,ai.getAttributeName(j)->cstr());
                int irr=1;

		double min = ptr[j];
		double max = ptr[j+tReal->sizeBounds];
		double size = ai.getSizeDomain(j);

		double sizeD=max-min;
		if(min<max && sizeD<size) {
			irr=0;
			sprintf(tmp1, "%f", min);
			sprintf(tmp2, "%f", max);
			sprintf(temp2, "[%s,%s]", tmp1, tmp2);
			strcat(temp,temp2);
		}
		if(!irr) {
			strcat(string, temp);
			strcat(string, "|");
		}
	}

	int cl=(int)ptr[tReal->ruleSize-1];
	sprintf(temp, "%s\n", ai.getNominalValue(tGlobals->numAttributesMC,cl)->cstr());
	strcat(string, temp);
}

void classifier_hyperrect_sse::crossover_1px(classifier_hyperrect_sse * in1,
				    classifier_hyperrect_sse * in2,
				    classifier_hyperrect_sse * out1,
				    classifier_hyperrect_sse * out2)
{
	out1->modif = out2->modif = 1;

	out1->length = tReal->ruleSize;
	out2->length = tReal->ruleSize;
	out1->chromosome = new aligned_float[tReal->ruleSize];
	out2->chromosome = new aligned_float[tReal->ruleSize];

	int cutPoint = rnd(0, tGlobals->numAttributesMC*2);
	int att=cutPoint/2;
	int value=cutPoint%2;


	bcopy(in1->chromosome,
	      out1->chromosome,
	      att * sizeof(float));
	bcopy(in2->chromosome,
	      out2->chromosome,
	      att * sizeof(float));
	bcopy(&in1->chromosome[tReal->sizeBounds],
	      &out1->chromosome[tReal->sizeBounds],
	      att * sizeof(float));
	bcopy(&in2->chromosome[tReal->sizeBounds],
	      &out2->chromosome[tReal->sizeBounds],
	      att * sizeof(float));

	if(value) {
		out1->chromosome[att]=in1->chromosome[att];
		out2->chromosome[att]=in2->chromosome[att];
		out1->chromosome[att+tReal->sizeBounds]=in2->chromosome[att+tReal->sizeBounds];
		out2->chromosome[att+tReal->sizeBounds]=in1->chromosome[att+tReal->sizeBounds];
		att++;
	}

	bcopy(&in1->chromosome[att],
	      &out2->chromosome[att],
	      (tReal->sizeBounds-att) * sizeof(float));
	bcopy(&in2->chromosome[att],
	      &out1->chromosome[att],
	      (tReal->sizeBounds-att) * sizeof(float));
	bcopy(&in1->chromosome[att+tReal->sizeBounds],
	      &out2->chromosome[att+tReal->sizeBounds],
	      (tReal->sizeBounds-att) * sizeof(float));
	bcopy(&in2->chromosome[att+tReal->sizeBounds],
	      &out1->chromosome[att+tReal->sizeBounds],
	      (tReal->sizeBounds-att) * sizeof(float));

	out1->chromosome[tReal->ruleSize-1]=in2->chromosome[tReal->ruleSize-1];
	out2->chromosome[tReal->ruleSize-1]=in1->chromosome[tReal->ruleSize-1];
}

void classifier_hyperrect_sse::postprocess()
{
}

