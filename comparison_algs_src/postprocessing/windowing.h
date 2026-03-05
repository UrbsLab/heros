#ifndef _WINDOWING_H_
#define _WINDOWING_H_

#include "instance.h"

class populationWrapper;

class windowing {
public:
	virtual void setInstances(instance **set,int howMuch)=0;
	virtual void newIteration(instance **&selectedInstances,int &howMuch, int &strataOffset)=0;
	virtual instance** getStratas()=0;
	virtual int numVersions(){return 1;}
	virtual int getCurrentVersion(){return 0;}
	virtual int needReEval(){return 1;}
};

#endif
