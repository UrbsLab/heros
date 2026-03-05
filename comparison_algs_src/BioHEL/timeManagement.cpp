#include "timeManagement.h"

#include <sys/time.h>
#include <sys/times.h>
#include <unistd.h>
#include <iostream>
#include <cstdio>
#include <errno.h>
#include "messageBuffer.h"

extern messageBuffer mb;

using namespace std;

timeManagement::timeManagement()
{
	initialTimeT=currentTimeT();
	initialTimeG=currentTimeG();
}

void timeManagement::resetTime()
{
	initialTimeT=currentTimeT();
}

timeManagement::~timeManagement()
{
        mb.printf("Total time: %g %g\n",totalTimeT(),totalTimeG());
}

double timeManagement::totalTime() 
{
	return currentTimeT();
}

double timeManagement::totalTimeT() 
{
	return currentTimeT()-initialTimeT;
}

double timeManagement::totalTimeG() 
{
	return currentTimeG()-initialTimeG;
}

double timeManagement::currentTimeT()
{
	struct tms cpu_time;
        times(&cpu_time);
	return (double)(cpu_time.tms_utime+cpu_time.tms_stime)
		/(double)sysconf(_SC_CLK_TCK);
}

double timeManagement::currentTimeG()
{
	struct timeval tv;

	if(gettimeofday(&tv,NULL)==-1) {
		perror("gettimeoday failed");
		exit(1);
	}
	return (double)tv.tv_sec+(double)tv.tv_usec/1000000.0;
}



