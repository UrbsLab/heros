#ifndef _CROSSOVER_DEF_
#define _CROSSOVER_DEF_

void crossover();
void crossSmart(int *parents,int numParents,int son);
void crossTwoParents(int parent1,int parent2,int son1,int son2);
void crossOneParent(int parent,int son);

#endif
