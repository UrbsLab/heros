#ifndef _INSTANCE_SET_
#define _INSTANCE_SET_

#include "classifier.h"
#include <stdio.h>
#include "JVector.h"
#include "windowing.h"
#include "sampling.h"

#define TRAIN 1
#define TEST 2

class instance;


class instanceSet {
private:
	Sampling **initSamplings;
	int *countsByClass;
	int **instByClass;
	void initInstanceLists();
	int numClasses;
	int classWiseInit;

	windowing *win;
	instance **set;
	instance **origSet;
	int numInstances;
	int numInstancesOrig;

	int strataOffset;

	int windowingEnabled;
	instance **window;
	int windowSize;

	void readFile(char fileName[],int &numInstances,int traintest);
	void parseHeader(FILE *,int traintest);
	void parseRelation(char *string);
	void parseReal(int numAtr);
	void parseInteger(char *string,int numAtr);
	void parseNominal(char *string,int numAtr);
	void parseAttribute(char *string,int numAtr);
	void initializeWindowing(int traintest);

public:
	instanceSet(char *fileName,int traintest);
	~instanceSet();

	inline instance *getInstance(int index) {
		if(windowingEnabled) return window[index];
		return set[index];
	}
	inline instance **getInstancesOfIteration() {
		if(windowingEnabled) return window;
		return set;
	}

	inline instance **getAllInstances() {
		return set;
	}

	inline instance **getOrigInstances() {
		return origSet;
	}
	inline int getNumInstancesOrig() {
		return numInstancesOrig;
	}

	inline instance **getStratas() {
		return win->getStratas();
	}

	instance *getInstanceInit(int forbiddenCL);

	inline int isWindowingEnabled() {return windowingEnabled;}
	int getNumInstancesOfIteration();
	int getStrataOffsetOfIteration();
	int newIteration(int isLast);
	inline int getNumInstances(){return numInstances;}
	int numVersions(){
		if(windowingEnabled) return win->numVersions();
		return 1;
	}
	int getCurrentVersion(){
		if(windowingEnabled) return win->getCurrentVersion();
		return 0;
	}

	void removeInstancesAndRestart(classifier *cla);
	void restart();

	int getMajorityClass();
	int getMajorityClassExcept(int cl);

	
};



#endif
