#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "JVector.h"

double getAverage(JVector<double> &vect)
{
	double ave=0;
	int i,size=vect.size();

	for(i=0;i<size;i++) ave+=vect[i];
	ave/=(double)size;
	return ave;
}

double getDeviation(JVector<double> &vect)
{
	double ave=getAverage(vect),dev=0;
	int i,size=vect.size();

	for(i=0;i<size;i++) dev+=pow(vect[i]-ave,2.0);
	dev/=(double)size;
	return sqrt(dev);
}
