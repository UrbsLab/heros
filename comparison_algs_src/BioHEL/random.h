#if !defined(_RANDOM_GEN)
#define _RANDOM_GEN

#include <stdlib.h>
#include <stdio.h>
//#include "mt19937ar-cok.h"
#include "mtwist.h"
#include "messageBuffer.h"

extern messageBuffer mb;

class Random {
      private:
	unsigned long int seed;
	int count1,count2;
	mt_prng mt;

	void buildSeed();

      public:
	 Random();
	~Random();
	void dumpSeed();
	unsigned long int getSeed() {return seed;}
	void setSeed( unsigned long int seed);
	unsigned long int getUInt();
	double operator ! (void);
	unsigned long int
	    operator() (unsigned long int uLow, unsigned long int uHigh);
	void dumpCounters() { mb.printf("Random stats %d %d\n",count1,count2);}
};

#endif
