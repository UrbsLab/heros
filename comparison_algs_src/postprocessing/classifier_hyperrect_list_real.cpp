#include "classifier_hyperrect_list_real.h"
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

double classifier_hyperrect_list_real::computeTheoryLength()
{
	int i,j,base;
	theoryLength = 0.0;

	float *ptr=predicates;

	for(i=0;i<numAtt;i++) {
		int att=whichAtt[i];
		float size=ai.getSizeDomain(att);
		if(size>0) {
			theoryLength += 1.0 - (ptr[1]-ptr[0])/size;
		}
		ptr+=tReal->attributeSize[att];
	}
	theoryLength/=(double)tGlobals->numAttributesMC;

	return theoryLength;
}

classifier_hyperrect_list_real::classifier_hyperrect_list_real()
{
	classifier_hyperrect_list_real::initializeChromosome();
}

classifier_hyperrect_list_real::classifier_hyperrect_list_real(const classifier_hyperrect_list_real & orig, int son)
{
	*this = orig;

	if (!son) {
		whichAtt = new int[numAtt];
		bcopy(orig.whichAtt, whichAtt, numAtt * sizeof(int));

		offsetPredicates = new int[numAtt];
		bcopy(orig.offsetPredicates, offsetPredicates, numAtt * sizeof(int));

		predicates = new float[ruleSize];	
		bcopy(orig.predicates, predicates, ruleSize * sizeof(float));
	} else {
		whichAtt = NULL;
		predicates = NULL;
		offsetPredicates=NULL;
	}
}

classifier_hyperrect_list_real::~classifier_hyperrect_list_real()
{
	delete whichAtt;
	delete predicates;
	delete offsetPredicates;
}

void classifier_hyperrect_list_real::initializeChromosome()
{
	int i,j,base;

	instance *ins=NULL;
	if(tGlobals->smartInit) {
		if(tGlobals->defaultClassPolicy!=DISABLED) {
			ins=is->getInstanceInit(tGlobals->defaultClass);
		} else {
			ins=is->getInstanceInit(ai.getNumClasses());
		}
	}

	JVector<int> selectedAtts;
	ruleSize=0;
	for(i=0;i<tGlobals->numAttributesMC;i++) {
		if(!rnd>=tReal->probIrr) {
			selectedAtts.addElement(i);
			ruleSize+=tReal->attributeSize[i];
		}
	}

	numAtt=selectedAtts.size();
	whichAtt = new int[numAtt];
	offsetPredicates = new int[numAtt];
	predicates = new float[ruleSize];

	for(i=0,base=0;i<numAtt;i++) {
		offsetPredicates[i]=base;
		int att=selectedAtts[i];
		whichAtt[i]=att;

		float max,min;
		float sizeD=ai.getSizeDomain(att);
		float minD=ai.getMinDomain(att);
		float maxD=ai.getMaxDomain(att);
		float size=(!rnd*0.5+0.25)*sizeD;

		if(ins) {
			float val=ins->realValues[att];
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

		predicates[base]=min;
		predicates[base+1]=max;

		base+=tReal->attributeSize[att];
	}

	if(ins) {
		classValue=ins->getClass();
	} else {
		do {
			classValue=rnd(0,ai.getNumClasses()-1);
		} while(tGlobals->defaultClassPolicy!=DISABLED && classValue==tGlobals->defaultClass);
	}
}

void classifier_hyperrect_list_real::crossover(classifier * in,
				classifier * out1, classifier * out2)
{
	crossover_1px(this, (classifier_hyperrect_list_real *) in,
		      (classifier_hyperrect_list_real *) out1,
		      (classifier_hyperrect_list_real *) out2);
}

float classifier_hyperrect_list_real::mutationOffset(float geneValue, float offsetMin,
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

void classifier_hyperrect_list_real::mutation()
{
	int i;
	int attribute, value,attIndex;

	modif = 1;

	if(tGlobals->numClasses>1 && !rnd<0.10) {
		int newValue;
		do {
			newValue = rnd(0, ai.getNumClasses()-1);
		} while (newValue == classValue || tGlobals->defaultClassPolicy!=DISABLED && newValue==tGlobals->defaultClass);
		classValue=newValue;
	} else {
		if(numAtt>0) {
			attIndex=rnd(0,numAtt-1);
			attribute=whichAtt[attIndex];
			value=rnd(0,tReal->attributeSize[attribute]-1);
			int index=offsetPredicates[attIndex]+value;

			float newValue,minOffset,maxOffset;
			minOffset = maxOffset = 0.5 * ai.getSizeDomain(attribute);
			newValue = mutationOffset(predicates[index], minOffset, maxOffset);
			if (newValue < ai.getMinDomain(attribute)) newValue = ai.getMinDomain(attribute);
			if (newValue > ai.getMaxDomain(attribute)) newValue = ai.getMaxDomain(attribute);
			predicates[index]=newValue;
			if(value) index--;
			if(predicates[index]>predicates[index+1]) {
				swapD(predicates[index],predicates[index+1]);
			}
		}
	}
}

void classifier_hyperrect_list_real::dumpPhenotype(char *string)
{
	char temp[10000];
	char temp2[10000];
	char tmp1[20];
	char tmp2[20];
	int i,j,index;

	strcpy(string, "");
	for (i = 0,index=0; i < numAtt; i++) {
		int attIndex=whichAtt[i];
		sprintf(temp,"Att %s is " ,ai.getAttributeName(attIndex)->cstr());
		int irr=1;

		float minD=ai.getMinDomain(attIndex);
		float maxD=ai.getMaxDomain(attIndex);
		if(predicates[index]==minD) {
			if(predicates[index+1]==maxD) {
				// do nothing
			} else {
				irr=0;
				sprintf(temp2, "[<%f]", predicates[index+1]);
				strcat(temp,temp2);
			}
		} else {
			if(predicates[index+1]==maxD) {
				irr=0;
				sprintf(temp2, "[>%f]", predicates[index]);
				strcat(temp,temp2);
			} else {
				irr=0;
				sprintf(tmp1, "%f", predicates[index]);
				sprintf(tmp2, "%f", predicates[index+1]);
				sprintf(temp2, "[%s,%s]", tmp1, tmp2);
				strcat(temp,temp2);
			}
		}

		index+=tReal->attributeSize[attIndex];

		if(!irr) {
			strcat(string, temp);
			strcat(string, "|");
		}
	}
	sprintf(temp, "%s\n", ai.getNominalValue(tGlobals->numAttributesMC,classValue)->cstr());
	strcat(string, temp);
}

void classifier_hyperrect_list_real::crossover_1px(classifier_hyperrect_list_real * in1,
				    classifier_hyperrect_list_real * in2,
				    classifier_hyperrect_list_real * out1,
				    classifier_hyperrect_list_real * out2)
{
	int i;

	out1->modif = out2->modif = 1;

	if(in1->numAtt==0) {
		classifier_hyperrect_list_real *tmp=in2;
		in2=in1;
		in1=tmp;
	}

	if(in1->numAtt==0) {
		out1->whichAtt=new int[out1->numAtt];
		out2->whichAtt=new int[out2->numAtt];
		out1->offsetPredicates=new int[out1->numAtt];
		out2->offsetPredicates=new int[out2->numAtt];
		out1->predicates=new float[out1->ruleSize];
		out2->predicates=new float[out2->ruleSize];
		return;
	}

	int pos1=rnd(0,in1->numAtt-1);
	int selAtt1=in1->whichAtt[pos1];

	for(i=0;i<in2->numAtt && in2->whichAtt[i]<selAtt1;i++);
	int pos2=i;
	int selAtt2;
	if(pos2!=in2->numAtt) {
		selAtt2=in2->whichAtt[pos2];
	} else {
		selAtt2=-1;
	}

	out1->numAtt=pos1+1+(in2->numAtt-pos2);
	out2->numAtt=pos2+(in1->numAtt-pos1-1);
	if(selAtt1==selAtt2) {
		out1->numAtt--;
		out2->numAtt++;
	}

	out1->whichAtt=new int[out1->numAtt];
	out2->whichAtt=new int[out2->numAtt];
	out1->offsetPredicates = new int[out1->numAtt];
	out2->offsetPredicates = new int[out2->numAtt];

	out1->ruleSize=0;
	for(i=0;i<=pos1;i++) {
		out1->whichAtt[i]=in1->whichAtt[i];
		out1->offsetPredicates[i]=out1->ruleSize;
		out1->ruleSize+=tReal->attributeSize[out1->whichAtt[i]];
	}
	int lenp1c1=out1->ruleSize;
	int base=pos2;
	if(selAtt1==selAtt2) {
		base++;
	}
	for(;i<out1->numAtt;i++,base++) {
		out1->whichAtt[i]=in2->whichAtt[base];
		out1->offsetPredicates[i]=out1->ruleSize;
		out1->ruleSize+=tReal->attributeSize[out1->whichAtt[i]];
	}

	out2->ruleSize=0;
	for(i=0;i<pos2;i++) {
		out2->whichAtt[i]=in2->whichAtt[i];
		out2->offsetPredicates[i]=out2->ruleSize;
		out2->ruleSize+=tReal->attributeSize[out2->whichAtt[i]];
	}
	int lenp2c1=out2->ruleSize;
	base=pos1;
	if(selAtt1!=selAtt2) {
		base++;
	}
	for(;i<out2->numAtt;i++,base++) {
		out2->whichAtt[i]=in1->whichAtt[base];
		out2->offsetPredicates[i]=out2->ruleSize;
		out2->ruleSize+=tReal->attributeSize[out2->whichAtt[i]];
	}

	out1->predicates=new float[out1->ruleSize];
	out2->predicates=new float[out2->ruleSize];

	bcopy(in1->predicates,out1->predicates,lenp1c1*sizeof(float));
	bcopy(in2->predicates,out2->predicates,lenp2c1*sizeof(float));

	if(selAtt1==selAtt2) {
		int baseP1=in1->offsetPredicates[pos1];
		int baseP2=in2->offsetPredicates[pos2];
		int baseO1=out1->offsetPredicates[pos1];
		int baseO2=out2->offsetPredicates[pos2];

		int cutPoint=rnd(0,2);
		if(cutPoint==0) {
			out1->predicates[baseO1]=in2->predicates[baseP2];
			out1->predicates[baseO1+1]=in2->predicates[baseP2+1];
			out2->predicates[baseO2]=in1->predicates[baseP1];
			out2->predicates[baseO2+1]=in1->predicates[baseP1+1];
		} else if(cutPoint==1) {
			float min1=in1->predicates[baseP1];
			float min2=in2->predicates[baseP2];
			float max1=in2->predicates[baseP2+1];
			float max2=in1->predicates[baseP1+1];

			if(min1>max1) swapD(min1,max1);
			if(min2>max2) swapD(min2,max2);

			out1->predicates[baseO1]=min1;
			out1->predicates[baseO1+1]=max1;
			out2->predicates[baseO2]=min2;
			out2->predicates[baseO2+1]=max2;
		} else {
			out1->predicates[baseO1]=in1->predicates[baseP1];
			out1->predicates[baseO1+1]=in1->predicates[baseP1+1];
			out2->predicates[baseO2]=in2->predicates[baseP2];
			out2->predicates[baseO2+1]=in2->predicates[baseP2+1];
		}
		pos2++;
	} else {
		int base1=in1->offsetPredicates[pos1];
		out1->whichAtt[pos1]=selAtt1;
		bcopy(&in1->predicates[base1],&out1->predicates[base1],tReal->attributeSize[selAtt1]*sizeof(float));
	}

	pos1++;
	if(pos1<in1->numAtt) {
		bcopy(&in1->predicates[in1->offsetPredicates[pos1]]
			,&out2->predicates[out2->offsetPredicates[pos2]]
			,(in1->ruleSize-in1->offsetPredicates[pos1])*sizeof(float));
	}
	if(pos2<in2->numAtt) {
		bcopy(&in2->predicates[in2->offsetPredicates[pos2]]
			,&out1->predicates[out1->offsetPredicates[pos1]]
			,(in2->ruleSize-in2->offsetPredicates[pos2])*sizeof(float));
	}


	if(!rnd<0.5) {
		out1->classValue=in1->classValue;
		out2->classValue=in2->classValue;
	} else {
		out1->classValue=in2->classValue;
		out2->classValue=in1->classValue;
	}
}

void classifier_hyperrect_list_real::postprocess()
{
}

void classifier_hyperrect_list_real::doSpecialStage(int stage)
{
	int i;

	if(stage==0) { //Generalize
		if(numAtt>0 && !rnd<tReal->probGeneralizeList) {
			int attribute=rnd(0,numAtt-1);
			int deletedSize=tReal->attributeSize[whichAtt[attribute]];

			int *newWhichAtt = new int[numAtt-1];
			int *newOffsetPredicates = new int[numAtt-1];
			float *newPredicates = new float[ruleSize-deletedSize];

			bcopy(whichAtt,newWhichAtt,attribute*sizeof(int));
			bcopy(offsetPredicates,newOffsetPredicates,attribute*sizeof(int));
			bcopy(predicates,newPredicates,offsetPredicates[attribute]*sizeof(float));

			if(attribute!=numAtt-1) {
				bcopy(&whichAtt[attribute+1],&newWhichAtt[attribute],(numAtt-attribute-1)*sizeof(int));
				bcopy(&offsetPredicates[attribute+1],&newOffsetPredicates[attribute]
					,(numAtt-attribute-1)*sizeof(int));
				bcopy(&predicates[offsetPredicates[attribute+1]],&newPredicates[offsetPredicates[attribute]]
					,(ruleSize-offsetPredicates[attribute+1])*sizeof(float));
			}


			delete whichAtt;
			whichAtt = newWhichAtt;
			delete offsetPredicates;
			offsetPredicates = newOffsetPredicates;
			delete predicates;
			predicates = newPredicates;
			numAtt--;
			ruleSize-=deletedSize;

			for(i=attribute;i<numAtt;i++) {
				offsetPredicates[i]-=deletedSize;
			}
		}
	} else { //Specialize
		if(numAtt < tGlobals->numAttributesMC && !rnd<tReal->probSpecializeList) {
			int attMap[tGlobals->numAttributesMC];
			bzero(attMap,tGlobals->numAttributesMC*sizeof(int));
			for(i=0;i<numAtt;i++) {
				attMap[whichAtt[i]]=1;
			}

			int selectedAtt;
			do {
				selectedAtt=rnd(0,tGlobals->numAttributesMC-1);
			} while(attMap[selectedAtt]==1);

			int addedSize=tReal->attributeSize[selectedAtt];
			int *newWhichAtt = new int[numAtt+1];
			int *newOffsetPredicates = new int[numAtt+1];
			float *newPredicates = new float[ruleSize+addedSize];

			int index=0,index2=0;
			while(index<numAtt && whichAtt[index]<selectedAtt) {
				newWhichAtt[index]=whichAtt[index];
				newOffsetPredicates[index]=offsetPredicates[index];
				bcopy(&predicates[index2],&newPredicates[index2]
					,tReal->attributeSize[whichAtt[index]]*sizeof(float));
				index2+=tReal->attributeSize[whichAtt[index]];
				index++;
			}
			newWhichAtt[index]=selectedAtt;
			newOffsetPredicates[index]=index2;

			float sizeD=ai.getSizeDomain(selectedAtt);
			float minD=ai.getMinDomain(selectedAtt);
			float maxD=ai.getMaxDomain(selectedAtt);
			float size=(!rnd*0.5+0.25)*sizeD;
			float min=!rnd*(sizeD-size)+minD;
			float max=min+size;
			newPredicates[index2]=min;
			newPredicates[index2+1]=max;

			if(index!=numAtt) {
				bcopy(&whichAtt[index],&newWhichAtt[index+1],(numAtt-index)*sizeof(int));
				bcopy(&offsetPredicates[index],&newOffsetPredicates[index+1],(numAtt-index)*sizeof(int));
				bcopy(&predicates[index2],&newPredicates[index2+addedSize]
					,(ruleSize-offsetPredicates[index])*sizeof(float));
			}

			delete whichAtt;
			whichAtt= newWhichAtt;
			delete offsetPredicates;
			offsetPredicates = newOffsetPredicates;
			delete predicates;
			predicates = newPredicates;
			numAtt++;
			ruleSize+=addedSize;

			for(i=index+1;i<numAtt;i++) {
				offsetPredicates[i]+=addedSize;
			}
		}
	}


}

