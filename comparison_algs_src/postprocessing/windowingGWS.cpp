#include "populationWrapper.h"
#include "windowingGWS.h"
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

void windowingGWS::setInstances(instance **pSet,int pHowMuch)
{
	int i;

	set=pSet;
	howMuch=pHowMuch;

	numClasses=ai.getNumClasses();
	numStrata=(int)round(cm.getParameter(WINDOWING_GWS));
	instancesOfClass= new instance **[numClasses];
	classSizes = new int[numClasses];
	classQuota = new double[numClasses];

	int capacity=0;
	for(i=0;i<numClasses;i++)  {
		int num=ai.getInstancesOfClass(i);
		instancesOfClass[i] = new instance *[num];
		classSizes[i]=0;
		classQuota[i]=(double)num/(double)numStrata;
		capacity+=(int)ceil(classQuota[i]);
	}
	sample = new instance *[capacity];

	for(i=0;i<howMuch;i++) {
		int cls=set[i]->getClass();
		instancesOfClass[cls][classSizes[cls]++]=set[i];
	}

	currentIteration=0;
}

windowingGWS::~windowingGWS()
{
	int i;

	delete classSizes;
	delete classQuota;
	for(i=0;i<numClasses;i++) delete instancesOfClass[i];
	delete instancesOfClass;
	delete sample;
}

void windowingGWS::newIteration(instance **&selectedInstances,int &numSelected, int &strataOffset)
{
	int i,j;

	stratum=currentIteration%numStrata;
	currentIteration++;

	sampleSize=0;
	for(i=0;i<numClasses;i++) {
		int fixQ=(int)classQuota[i];
		for(j=0;j<fixQ;j++) {
			int pos=rnd(0,classSizes[i]-1);
			sample[sampleSize++]=instancesOfClass[i][pos];
		}
		double prob=classQuota[i]-fixQ;
		if(!rnd<prob) {
			int pos=rnd(0,classSizes[i]-1);
			sample[sampleSize++]=instancesOfClass[i][pos];
		}
	}

	numSelected=sampleSize;
	selectedInstances=sample;
}

