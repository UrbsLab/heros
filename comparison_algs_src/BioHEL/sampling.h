#ifndef _SAMPLING_H_
#define _SAMPLING_H_

#include "random.h"

extern Random rnd;

class Sampling {
	int maxSize;
	int num;
	int *sample;

	void initSampling() {
		int i;
		for(i=0;i<maxSize;i++) sample[i]=i;
		num=maxSize;
	}

public:
	Sampling(int pMaxSize) {
		maxSize=pMaxSize;
		sample = new int[maxSize];
		initSampling();
	}

	~Sampling() {
		delete [] sample;
	}

	int getSample() {
		int pos=rnd(0,num-1);
		int value=sample[pos];
		sample[pos]=sample[num-1];
		num--;

		if(num==0) initSampling();

		return value;
	}

	int numSamplesLeft() {return num;}
};

#endif
