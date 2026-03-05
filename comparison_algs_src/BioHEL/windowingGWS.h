#ifndef _WINDOWING_GWS_H_
#define _WINDOWING_GWS_H_

#include "windowing.h"
#include "JVector.h"

class windowingGWS : public windowing{
	instance **set;
	instance ***instancesOfClass;
	instance **sample;
	int sampleSize;
	double *classQuota;
	int *classSizes;
	int howMuch;
	int numStrata;
	int numClasses;
	int stratum;
	int currentIteration;

public:
	~windowingGWS();
	void setInstances(instance **set,int howMuch);
	void newIteration(instance ** &selectedInstances,int &howMuch, int &strataOffset);
	int needReEval() {
		return 1;
	}

	int numVersions() {
		return numStrata;
	}

	int getCurrentVersion() {
		return stratum;
	}
	
	instance ** getStratas() {
		return set;
	}
};

#endif
