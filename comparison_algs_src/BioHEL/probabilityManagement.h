#ifndef _PROBABILITY_MANAGEMENT_H_
#define _PROBABILITY_MANAGEMENT_H_

#include <math.h>

#define LINEAR 1
#define SIGMOIDAL 2

extern double percentageOfLearning;

class probabilityManagement {
	double probStart;
	double probEnd;
	double probLength;
	int evolMode;

	double currentProb;
	double sigmaYLength;
	double sigmaYBase;
	double sigmaXOffset;
	double beta;
      public:
	 probabilityManagement(double start, double end, int mode) {
		probStart = start;
		probEnd = end;
		evolMode = mode;

		if (mode == LINEAR) {
			probLength=end-start;
			currentProb = start;
		} else {
			sigmaYLength = end - start;
			sigmaYBase = start;
			sigmaXOffset = 0.5;
			beta = -10;
		}
	}

	inline double incStep() {
		if (evolMode == LINEAR) {
			currentProb=percentageOfLearning*probLength+probStart;
		} else {
			currentProb = 
			    sigmaYLength / (1 +
					    exp(beta*(percentageOfLearning-0.5)))
				  + sigmaYBase;
			
		}
		return currentProb;
	}
};

#endif
