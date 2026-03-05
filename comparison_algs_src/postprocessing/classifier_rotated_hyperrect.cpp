#include "classifier_rotated_hyperrect.h"
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

double classifier_rotated_hyperrect::computeTheoryLength()
{
	int i, j, k;
	float *ptr = chromosome;
	theoryLength = 0.0;

	for (j = 0; j < tGlobals->numAttributesMC; j++) {
                float vmin=ptr[j];
                float vmax=ptr[tReal->sizeBounds+j];
		if(vmin<=vmax) {
			theoryLength += 1.0 - (vmax-vmin)/tReal->sizeD[j];
		}
	}

	theoryLength/=(double)tGlobals->numAttributesMC;

	double defAngles=0;
	int base=tReal->sizeBounds*2+1;
	for(i=0;i<tReal->numUsedAngles;i++,base+=2) {
		if(ptr[base]!=tReal->step0) defAngles++;
	}
	defAngles/=(double)tReal->numUsedAngles;
	theoryLength+=defAngles*0.05;

	return theoryLength;
}

classifier_rotated_hyperrect::classifier_rotated_hyperrect()
{
	length = tReal->ruleSize;
	chromosome = new float[tReal->ruleSize];
	classifier_rotated_hyperrect::initializeChromosome();
}

classifier_rotated_hyperrect::classifier_rotated_hyperrect(const classifier_rotated_hyperrect & orig, int son)
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

classifier_rotated_hyperrect::~classifier_rotated_hyperrect()
{
	delete chromosome;
}

void classifier_rotated_hyperrect::initializeChromosome()
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

	int index=tReal->sizeBounds*2;
	int attMask[tGlobals->numAttributesMC];
	bzero(attMask,sizeof(int)*tGlobals->numAttributesMC);
	int angleMask[tReal->numAngles];
	int countRotAtt=0;
	bzero(angleMask,sizeof(int)*tReal->numAngles);
	for(j=0;j<tReal->numUsedAngles;j++,index+=2) {
		int angle;
		do {
			angle=rnd(0,tReal->numAngles-1);
		} while(angleMask[angle]);
		angleMask[angle]=1;

		if(attMask[tReal->angleList1[angle]]==0) countRotAtt++;	
		attMask[tReal->angleList1[angle]]=1;
		if(attMask[tReal->angleList2[angle]]==0) countRotAtt++;	
		attMask[tReal->angleList2[angle]]=1;

		chromosome[index]=angle;
		if(!rnd<tReal->prob0AngleInit) {
			chromosome[index+1]=tReal->step0;
		} else {
			chromosome[index+1]=rnd(0,tReal->numSteps-1);
		}
	}

	int relAttIndex=0;
	int rotAtts[countRotAtt];
	for(i=0;i<tGlobals->numAttributesMC;i++) {
		if(attMask[i]==1) {
			rotAtts[relAttIndex++]=i;
		}
	}
	relAttIndex=0;
	
	double probIrr=tReal->probIrr+(double)countRotAtt/(double)tGlobals->numAttributesMC;

	
	float realValues[tGlobals->numAttributesMC];
	if(ins) {
		bcopy(ins->realValues,realValues,sizeof(float)*tGlobals->numAttributesMC);	
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
	
	}


	for (j = 0; j < tGlobals->numAttributesMC; j++) {
		double min,max;
		if(rotAtts[relAttIndex]!=j && !rnd<probIrr) {
			max=!rnd*tReal->sizeD[j]+tReal->minD[j];
			min=!rnd*(tReal->maxD[j]-max)+max;
		} else {
			if(rotAtts[relAttIndex]==j) relAttIndex++;
			double size=(!rnd*0.5+0.25)*tReal->sizeD[j];
			if(ins) {
				double val=realValues[j];
                                min=val-size/2.0;
                                max=val+size/2.0;
                                if(min<tReal->minD[j]) {
                                        max+=(tReal->minD[j]-min);
                                        min=tReal->minD[j];
                                }
                                if(max>tReal->maxD[j]) {
                                        min-=(max-tReal->maxD[j]);
                                        max=tReal->maxD[j];
                                }
			} else {
                                min=!rnd*(tReal->sizeD[j]-size)+tReal->minD[j];
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

void classifier_rotated_hyperrect::setGene(short int gene, short int value, double nfo)
{
	if(gene<tGlobals->numAttributesMC) {
		chromosome[gene + value*tReal->sizeBounds] = (float)nfo;
	} else {
		chromosome[tReal->ruleSize-1] = (float)nfo;
	}
	modif = 1;
}

double classifier_rotated_hyperrect::getGene(short int gene, short int value)
{
	if(gene<tGlobals->numAttributesMC) {
		return chromosome[gene + value*tReal->sizeBounds];
	} else {
		return chromosome[tReal->ruleSize-1];
	}
}

void classifier_rotated_hyperrect::crossover(classifier * in,
				classifier * out1, classifier * out2)
{
	crossover_1px(this, (classifier_rotated_hyperrect *) in,
		      (classifier_rotated_hyperrect *) out1,
		      (classifier_rotated_hyperrect *) out2);
}

double classifier_rotated_hyperrect::mutationOffset(double geneValue, double offsetMin,
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

void classifier_rotated_hyperrect::mutation()
{
	int i;
	int attribute=-1, value;

	modif = 1;

	int modifAngles=0;

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
		    0.5 * tReal->sizeD[attribute];
		newValue = mutationOffset(oldValue, minOffset,
					  maxOffset);
		if (newValue < tReal->minD[attribute])
			newValue = tReal->minD[attribute];
		if (newValue > tReal->maxD[attribute])
			newValue = tReal->maxD[attribute];
		setGene(attribute, value, newValue);
	} else {
		int newValue;
		int oldValue = (int) getGene(attribute, value);
		do {
			newValue = rnd(0, ai.getNumClasses()-1);
		} while (newValue == oldValue || tGlobals->defaultClassPolicy!=DISABLED && newValue==tGlobals->defaultClass);
		setGene(attribute, value, newValue);
	}

	int pos=tReal->sizeBounds*2+rnd(0,tReal->sizeAngles-1);
	if(pos%2) {
		if(!rnd<tReal->prob0AngleMut) {
			chromosome[pos]=tReal->step0;
		} else {
			int newValue=(int)chromosome[pos];
			if(!rnd<0.5) {
				newValue+=rnd(1,tReal->mutSteps);
				if(newValue>=tReal->numSteps) newValue=tReal->numSteps-1;
			} else {
				newValue-=rnd(1,tReal->mutSteps);
				if(newValue<0) newValue=0;
			}
			chromosome[pos]=newValue;
		}
	} else {
		int angleMask[tReal->numAngles];
		bzero(angleMask,tReal->numAngles*sizeof(int));

		chromosome[pos]=rnd(0,tReal->numAngles-1);
		int base=tReal->sizeBounds*2;
		for(i=0;i<tReal->numUsedAngles;i++,base+=2) {
			if(angleMask[(int)chromosome[base]]) {
				int pos;
				do {
					pos=rnd(0,tReal->numAngles-1);
				} while(angleMask[pos]);
				chromosome[base]=pos;
			}
			angleMask[(int)chromosome[base]]=1;
		}
	}
}

void classifier_rotated_hyperrect::dumpPhenotype(char *string)
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

                double size=max-min;
                if(min<max && size<tReal->sizeD[j]) {
                        irr=0;
			if(min>tReal->minD[j]) {
				if(max<tReal->maxD[j]) {
		                        sprintf(temp2, "[%f,%f]", min, max);
		                        strcat(temp,temp2);
				} else {
		                        sprintf(temp2, "[>%f]", min);
		                        strcat(temp,temp2);
				}
			} else {
	                        sprintf(temp2, "[<%f]", max);
	                        strcat(temp,temp2);
			}
                }

		if(!irr) {
			strcat(string, temp);
			strcat(string, "|");
		}
	}

	
	strcat(string,"|");
	int index=tReal->sizeBounds*2;
	for(i=0;i<tReal->numUsedAngles;i++,index+=2) {
		if(ptr[index+1]!=tReal->step0) {
			sprintf(temp,"Angle %s %s : %f|"
				,ai.getAttributeName(tReal->angleList1[(int)ptr[index]])->cstr()
				,ai.getAttributeName(tReal->angleList2[(int)ptr[index]])->cstr()
				,(ptr[index+1]-tReal->step0)*tReal->stepRatio);
			strcat(string,temp);
		}
	}

	int cl=(int)ptr[tReal->ruleSize-1];
	sprintf(temp, "%s\n", ai.getNominalValue(tGlobals->numAttributesMC,cl)->cstr());
	strcat(string, temp);
}

void classifier_rotated_hyperrect::crossover_1px(classifier_rotated_hyperrect * in1,
				    classifier_rotated_hyperrect * in2,
				    classifier_rotated_hyperrect * out1,
				    classifier_rotated_hyperrect * out2)
{
	int i;

	out1->modif = out2->modif = 1;

	out1->length = tReal->ruleSize;
	out2->length = tReal->ruleSize;
	out1->chromosome = new float[tReal->ruleSize];
	out2->chromosome = new float[tReal->ruleSize];

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


	int cutPoint2 = rnd(0,tReal->sizeAngles-1);
	int base=tReal->sizeBounds*2;
	bcopy(&in1->chromosome[base],
	      &out1->chromosome[base],
	      cutPoint2 * sizeof(float));
	bcopy(&in2->chromosome[base],
	      &out2->chromosome[base],
	      cutPoint2 * sizeof(float));
	bcopy(&in1->chromosome[base+cutPoint2],
	      &out2->chromosome[base+cutPoint2],
	      (tReal->sizeAngles-cutPoint2) * sizeof(float));
	bcopy(&in2->chromosome[base+cutPoint2],
	      &out1->chromosome[base+cutPoint2],
	      (tReal->sizeAngles-cutPoint2) * sizeof(float));

	int angleMask[tReal->numAngles];
	bzero(angleMask,tReal->numAngles*sizeof(int));
	for(i=0;i<tReal->numUsedAngles;i++,base+=2) {
		if(angleMask[(int)out1->chromosome[base]]) {
			int pos;
			do {
				pos=rnd(0,tReal->numAngles-1);
			} while(angleMask[pos]);
			out1->chromosome[base]=pos;
		}
		angleMask[(int)out1->chromosome[base]]=1;
	}

	bzero(angleMask,tReal->numAngles*sizeof(int));
	base=tReal->sizeBounds*2;
	for(i=0;i<tReal->numUsedAngles;i++,base+=2) {
		if(angleMask[(int)out2->chromosome[base]]) {
			int pos;
			do {
				pos=rnd(0,tReal->numAngles-1);
			} while(angleMask[pos]);
			out2->chromosome[base]=pos;
		}
		angleMask[(int)out2->chromosome[base]]=1;
	}
}

void classifier_rotated_hyperrect::postprocess()
{
}

