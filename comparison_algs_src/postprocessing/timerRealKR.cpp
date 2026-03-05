#include "classifier_hyperrect_list.h"
#include "timerRealKR.h"
#include "timerMDL.h"
#include "attributesInfo.h"
#include "random.h"
#include "messageBuffer.h"
#include <math.h>
#include "timerGlobals.h"
#include "populationWrapper.h"

extern messageBuffer mb;

extern attributesInfo ai;
extern Random rnd;
extern timerMDL *tMDL;
extern timerGlobals *tGlobals;

timerRealKR::timerRealKR()
{
	if (!cm.thereIsParameter(KR_HYPERRECT)) {
		enabled = 0;
		return;
	}
	enabled = 1;
	int i,j;

	tGlobals->probOne = cm.getParameter(PROB_ONE);

	int numAtt=ai.getNumAttributesMC();
	int numExpAtt=(int)cm.getParameter(EXPRESSED_ATT_INIT);
	if(numExpAtt>numAtt) numExpAtt=numAtt;
	probIrr= 1-(double)numExpAtt/(double)numAtt;
	mb.printf("Probability of irrelevant attribute set to %f\n",probIrr);

	hyperrectList=cm.thereIsParameter(HYPERRECT_LIST);
	if(hyperrectList) {
		probGeneralizeList=cm.getParameter(PROB_GENERALIZE_LIST);
		probSpecializeList=cm.getParameter(PROB_SPECIALIZE_LIST);
		//fitGen = new int [numAtt];
		//fitSpe = new int [numAtt];
	}

	if(!hyperrectList && ai.onlyRealValuedAttributes()) {
		mb.printf("Using SSE instructions for intervalar representation\n");
		attPerBlock=4;
		sizeBounds=numAtt;
		if(sizeBounds%4) {
			sizeBounds+=(4-sizeBounds%4);
		}
		ruleSize=sizeBounds*2+1;
		rotateIntervals=cm.thereIsParameter(ROTATE_HYPERRECTANGLES);
		if(rotateIntervals) {
			numSteps=64;
			step0=numSteps/2;
			mutSteps=numSteps/4;
			stepRatio=M_PI*2/numSteps;
			sinTable=new float[numSteps];
			cosTable=new float[numSteps];
			double angle=-M_PI;
			for(i=0;i<numSteps;i++,angle+=stepRatio) {
				sinTable[i]=sin(angle);
				cosTable[i]=cos(angle);
			}
			prob0AngleInit=cm.getParameter(PROB_0ANGLE_INIT);
			prob0AngleMut=cm.getParameter(PROB_0ANGLE_MUT);
			numUsedAngles=3;

			minD=new float[numAtt];
			maxD=new float[numAtt];
			sizeD=new float[numAtt];

			if(cm.thereIsParameter(RESTRICTED_ROTATED_ATTRIBUTES)) {
				int attMap[numAtt];
				char string[100];
				FILE *fp;

				for(i=0;i<numAtt;i++) attMap[i]=0;
				
				fp=fopen("rotatedAttributes.dat","r");
				if(fp==NULL) {
					fprintf(stderr,"Cannot open rotatedAttributes.dat\n");
					exit(1);
				}

				JVector<int> atts1;
				JVector<int> atts2;
				fgets(string,99,fp);
				while(!feof(fp)) {
					char *ptr=strtok(string," ");
					int att1=atoi(ptr);
					ptr=strtok(NULL," ");
					int att2=atoi(ptr);
					attMap[att1]++;
					attMap[att2]++;

					atts1.addElement(att1);
					atts2.addElement(att2);

					fgets(string,99,fp);
				}
				fclose(fp);

				numAngles=atts1.size();
				angleList1 = new int[numAngles];
				angleList2 = new int[numAngles];
				for(i=0;i<numAngles;i++) {
					angleList1[i]=atts1[i];
					angleList2[i]=atts2[i];
				}

				if(numUsedAngles>numAngles) 
					numUsedAngles=numAngles;

				for(i=0;i<numAtt;i++) {
					int num=0;
					if(attMap[i]) num=numUsedAngles;
					maxD[i]=sqrt(num+1);
					minD[i]=-maxD[i];
					sizeD[i]=2*maxD[i];
				}
			} else {
				numAngles=numAtt*(numAtt-1)/2;
				angleList1 = new int[numAngles];
				angleList2 = new int[numAngles];
				int index=0;
				for(i=0;i<numAtt-1;i++) {
					for(j=i+1;j<numAtt;j++,index++) {
						angleList1[index]=i;
						angleList2[index]=j;	
					}
				}

				if(numUsedAngles>numAngles) 
					numUsedAngles=numAngles;

				for(i=0;i<numAtt;i++) {
					maxD[i]=sqrt(numUsedAngles+1);
					minD[i]=-maxD[i];
					sizeD[i]=2*maxD[i];
				}
			}

			sizeAngles=numUsedAngles*2;
			ruleSize+=sizeAngles;
		}
	} else {
		attributeSize = new int[ai.getNumAttributes()];
		attributeOffset = new int[ai.getNumAttributes()];
		evaluators = new evaluator *[ai.getNumAttributesMC()];

		ruleSize = 0;
		for (i = 0; i < ai.getNumAttributesMC(); i++) {
			if (ai.getTypeOfAttribute(i) == REAL) {
				evaluators[i] = new evaluatorReal;
				attributeSize[i] = 2;
			} else {
				evaluators[i] = new evaluatorNominal;
				attributeSize[i] = ai.getNumValuesAttribute(i);
			}
			attributeOffset[i] = ruleSize;
			ruleSize += attributeSize[i];
		}
		attributeSize[i] = 1;
		attributeOffset[i] = ruleSize;
		ruleSize++;
	}


	thereIsSpecialCrossover = 0;
	if (cm.thereIsParameter(ALPHA_OF_BLX)) {
		alphaOfBLX = cm.getParameter(ALPHA_OF_BLX);
	} else
		alphaOfBLX = -1;

	if (cm.thereIsParameter(D_OF_FR)) {
		dOfFR = cm.getParameter(D_OF_FR);
	} else
		dOfFR = -1;

	if (cm.thereIsParameter(N_OF_SBX)) {
		nOfSBX = cm.getParameter(N_OF_SBX);
	} else
		nOfSBX = -1;

	if (alphaOfBLX != -1 || dOfFR != -1 || nOfSBX != -1)
		thereIsSpecialCrossover = 1;
}

void timerRealKR::dumpStats(int iteration)
{
	if (!enabled)
		return;
}

void timerRealKR::specialCrossover(float parent1, float parent2,
				   float &son1, float &son2,
				   float minDomain, float maxDomain)
{
	if (alphaOfBLX != -1)
		crossoverBLX(parent1, parent2, son1, son2, minDomain,
			     maxDomain);
	else if (dOfFR != -1)
		crossoverFR(parent1, parent2, son1, son2, minDomain,
			    maxDomain);
	else
		crossoverSBX(parent1, parent2, son1, son2, minDomain,
			     maxDomain);
}

void timerRealKR::crossoverBLX(float parent1, float parent2,
			       float &son1, float &son2,
			       float minDomain, float maxDomain)
{
	float minPare = fmin(parent1, parent2);
	float maxPare = fmax(parent1, parent2);
	float interval = maxPare - minPare;

	float minInterval = minPare - alphaOfBLX * interval;
	if (minInterval < minDomain)
		minInterval = minDomain;

	float maxInterval = maxPare + alphaOfBLX * interval;
	if (maxInterval > maxDomain)
		maxInterval = maxDomain;

	son1 = !rnd * (maxInterval - minInterval) + minInterval;
	son2 = !rnd * (maxInterval - minInterval) + minInterval;
}

void timerRealKR::crossoverSBX(float parent1, float parent2,
			       float &son1, float &son2,
			       float minDomain, float maxDomain)
{
	float u, beta;

	u = !rnd;
	u *= 0.9999999999;
	if (u <= 0.5) {
		beta = pow(2 * u, 1.0 / (nOfSBX + 1));
	} else {
		beta = pow(1.0 / (2.0 * (1 - u)), 1.0 / (nOfSBX + 1));
	}

	son1 = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2);
	if (son1 < minDomain)
		son1 = minDomain;
	if (son1 > maxDomain)
		son1 = maxDomain;

	son2 = 0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2);
	if (son2 < minDomain)
		son2 = minDomain;
	if (son2 > maxDomain)
		son2 = maxDomain;
}

void timerRealKR::crossoverFR(float parent1, float parent2, float &son1,
			      float &son2, float minDomain,
			      float maxDomain)
{
	float minPare = fmin(parent1, parent2);
	float maxPare = fmax(parent1, parent2);
	float interval = maxPare - minPare;

	son1 =
	    crossoverFRp(minPare, maxPare, interval, minDomain, maxDomain);
	son2 =
	    crossoverFRp(minPare, maxPare, interval, minDomain, maxDomain);
}

float timerRealKR::crossoverFRp(float minPare, float maxPare,
				 float interval, float minDomain,
				 float maxDomain)
{
	float centre;
	float valor;

	if (!rnd < 0.5) {
		centre = minPare;
	} else {
		centre = maxPare;
	}

	if (!rnd < 0.5) {
		valor = centre + (!rnd - 1) * dOfFR * interval;
	} else {
		valor = centre + (1 - !rnd) * dOfFR * interval;
	}
	if (valor < minDomain)
		valor = minDomain;
	if (valor > maxDomain)
		valor = maxDomain;

	return valor;
}

float timerRealKR::fmin(float a, float b)
{
	if (a < b)
		return a;
	return b;
}
float timerRealKR::fmax(float a, float b)
{
	if (a > b)
		return a;
	return b;
}

int rankOrderAtt(const void *pA, const void *pB)
{
	rankAtt *a=(rankAtt *)pA;
	rankAtt *b=(rankAtt *)pB;

	if(a->count>b->count) return -1;
	if(a->count<b->count) return +1;
	return 0;
}


void timerRealKR::newIteration(int iteration,int finalIteration)
{
	if (!enabled)
		return;

	/*if(hyperrectList) {
		int i,j;

		for(i=0;i<tGlobals->numAttributesMC;i++) {
			fitGen[i]=0;
		}

		int numExp=0;
		classifier **pop=pw->getPopulation();
		for(i=0;i<pw->popSize;i++) {
			classifier_hyperrect_list *ind=(classifier_hyperrect_list *)pop[i];
			for(j=0;j<ind->numAtt;j++) {
				int att=ind->whichAtt[j];
				if(!fitGen[att]) numExp++;
				fitGen[att]++;
			}
		}

		for(i=0;i<tGlobals->numAttributesMC;i++) {
			fitSpe[i]=pw->popSize-fitGen[i];
		}

		//rankAtt attData[tGlobals->numAttributesMC];
		//for(i=0;i<tGlobals->numAttributesMC;i++) {
		//	attData[i].pos=i;
		//	attData[i].count=fitGen[i];
		//}
		//qsort(attData, tGlobals->numAttributesMC,sizeof(rankAtt),rankOrderAtt);

		//mb.printf("It %d,Number of expressed attributes: %d\n",iteration,numExp);
		//mb.printf("Rank of attributes\n");
		//for(i=0;i<tGlobals->numAttributesMC;i++) {
		//	printf("Att %s : %d\n",ai.getAttributeName(attData[i].pos)->cstr(),attData[i].count);
		//}
	}*/
}
