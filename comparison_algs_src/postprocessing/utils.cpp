#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "JVector.h"

double getAverage(JVector<double> &vect)
{
    double ave=0;
    int i,size=vect.size();

    for(i=0; i<size; i++) ave+=vect[i];
    ave/=(double)size;
    return ave;
}

double getDeviation(JVector<double> &vect)
{
    double ave=getAverage(vect),dev=0;
    int i,size=vect.size();

    for(i=0; i<size; i++) dev+=pow(vect[i]-ave,2.0);
    dev/=(double)size;
    return sqrt(dev);
}


bool AlmostEqualRelative2(double A, double  B)
{
    double  maxRelativeError = 0.05; // their normalized difference (== error ) must be within 5%; e.g. true <=> A is in [b*0.95,b*1.05]

    if (A == B)
        return true;

    double  relativeError;
    if (fabs(B) > fabs(A))
        relativeError = fabs((A - B) / B);
    else
        relativeError = fabs((A - B) / A);
    if (relativeError <= maxRelativeError)
        return true;

    return false;
}
