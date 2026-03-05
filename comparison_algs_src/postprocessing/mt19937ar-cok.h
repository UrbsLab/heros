#ifndef _MT19937_H_
#define _MT19937_H_

void init_genrand(unsigned long s);
void init_by_array(unsigned long init_key[], unsigned long key_length);
//static void next_state(void);
unsigned long genrand_int32(void);
long genrand_int31(void);
double genrand_real1(void);
double genrand_real2(void);
double genrand_real3(void);
double genrand_res53(void);

#endif
