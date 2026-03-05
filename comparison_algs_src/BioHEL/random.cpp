#include <math.h>
#include <time.h>
#include "random.h"
#include <sys/utsname.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include "messageBuffer.h"

extern messageBuffer mb;

// ------------------------------------
// Constructor per defecte de la classe
// ------------------------------------

Random::Random()
{
	buildSeed();
	mt.seed32(seed);
	count1=count2=0;
	//init_genrand(seed);
}

Random::~Random()
{
}

void Random::buildSeed()
{
	FILE *fp;
	fp = fopen("/dev/urandom", "rb");
	if (!fp) {
		perror("random fopen");
		exit(1);
	}

	if (fread(&seed, sizeof(unsigned long int), 1, fp) != 1) {
		perror("random fread");
		exit(1);
	}
	fclose(fp);
}

void Random::dumpSeed()
{
	int i;
	mb.printf("Random seed %u\n",seed);
}

void Random::setSeed(unsigned long int pSeed)
{
	seed=pSeed;
	mt.seed32(seed);
	//init_genrand(seed);
}

double Random::operator !(void)
{
	//return genrand_real1();
	return mt.drand();
}

unsigned long int Random::operator() (unsigned long int uLow,
				      unsigned long int uHigh) {
	//return (uLow + (unsigned long int)(genrand_real2()*(uHigh + 1 - uLow)));
	return (uLow + (unsigned long int)(mt.drand()*(uHigh + 1 - uLow)));
}

unsigned long int Random::getUInt() 
{
	return mt.lrand();
}
