#ifndef _GESTIO_TEMPS_H_
#define _GESTIO_TEMPS_H_

#include "JString.h"


class timeManagement {
	double startTime;
	double timeInterval;
	double initialTimeT;
	double initialTimeG;

	double currentTimeT();
	double currentTimeG();
	double totalTimeG();
	double totalTimeT();
public:
	timeManagement();
	~timeManagement();
	double totalTime();
	void resetTime();
};

#endif
