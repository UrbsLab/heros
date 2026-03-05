#include "populationWrapper.h"
#include "windowingILAS.h"
#include "configManagement.h"
#include "attributesInfo.h"
#include "random.h"
#include "classifierFitness.h"
#include "messageBuffer.h"

extern messageBuffer mb;
extern Random rnd;
extern attributesInfo ai;
extern configManagement cm;
extern int nodeRank;

void windowingILAS::setInstances(instance **pSet,int pHowMuch)
{
	int i;

	if(strata!=NULL) {
		if(pHowMuch>howMuch) {
			delete strata;
			strata = new instance *[pHowMuch];
		}
	} else {
		strata = new instance *[pHowMuch];
	}

	set=pSet;
	howMuch=pHowMuch;
	currentIteration=0;
	reorderInstances();
}

windowingILAS::windowingILAS()
{
	numStrata=(int)round(cm.getParameter(WINDOWING_ILAS));
	strataSizes = new int[numStrata];
	strataOffsets = new int[numStrata];
	strata=NULL;
}

windowingILAS::~windowingILAS()
{
	delete strata;
	delete strataSizes;
	delete strataOffsets;
}

void windowingILAS::reorderInstances()
{
	int i,j,k;
	int nc=ai.getNumClasses();

	Sampling **samplings = new Sampling *[nc];
	for(i=0;i<nc;i++) {
		samplings[i] = new Sampling(numStrata);
	}

	int tempCapacity=howMuch/numStrata+nc;
	int *countTemp = new int[numStrata];
	instance ***tempStrata = new instance **[numStrata];
	for(i=0;i<numStrata;i++) {
		countTemp[i]=0;
		tempStrata[i]= new instance *[tempCapacity];
	}

	for(i=0;i<howMuch;i++) {
		int cls=set[i]->getClass();
		int str=samplings[cls]->getSample();
		tempStrata[str][countTemp[str]++]=set[i];
	}

	int acum=0;
	for(i=0;i<numStrata;i++) {
		int size=countTemp[i];
		strataSizes[i]=size;
		strataOffsets[i]=acum;
		for(j=0;j<size;j++) {
			strata[acum++]=tempStrata[i][j];
		}
	}

	for(i=0;i<numStrata;i++) {
		delete [] tempStrata[i];
	}
	delete [] countTemp;
	delete [] tempStrata;

	for(i=0;i<nc;i++) {
		delete samplings[i];
	}
	delete [] samplings;
}

void windowingILAS::newIteration(instance **&selectedInstances,int &numSelected, int &strataOffset)
{
	stratum=currentIteration%numStrata;
	numSelected=strataSizes[stratum];
	selectedInstances=&strata[strataOffsets[stratum]];
	strataOffset=strataOffsets[stratum];
	currentIteration++;
}

