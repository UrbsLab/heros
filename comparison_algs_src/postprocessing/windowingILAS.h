#ifndef _WINDOWING_ILAS_H_
#define _WINDOWING_ILAS_H_

#include "windowing.h"
#include "JVector.h"

class windowingILAS: public windowing {
	instance **set;
	instance **strata;
	int *strataSizes;
	int *strataOffsets;
	int howMuch;
	int numStrata;
	int currentIteration;
	int instancesPerStrata;
	int thereIsStatisticalValidation;
	int stratum;

	void reorderInstances();
	double nominalAttributeValidation(int pClass, int attr,
			JVector<instance *>&sample);
	double realValuedAttributeValidation(int pClass, int attr, JVector<
			instance *>&sample);
	double statisticalValidation(int pClass, JVector<instance *>&sample);

public:
	windowingILAS();
	~windowingILAS();
	void setInstances(instance **set, int howMuch);
	void newIteration(instance ** &selectedInstances, int &howMuch,
			int &strataOffset);
	int needReEval() {
		if (numStrata == 1)
			return 0;
		return 1;
	}
	int numVersions() {
		return numStrata;
	}
	int getCurrentVersion() {
		return stratum;
	}

	instance ** getStratas() {
		return strata;
	}

};

#endif
