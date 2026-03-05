#include "timerCrossover.h"
#include <math.h>
#include "messageBuffer.h"
#include "attributesInfo.h"

extern messageBuffer mb;
extern attributesInfo ai;

timerCrossover::timerCrossover()
{
	int i;

	cxOperator = (int) cm.getParameter(CROSSOVER_OPERATOR);
	crossoverProb = cm.getParameter(PROB_CROSSOVER);
	
	if(cxOperator == CROSS_INFORMED) {
		int numAtt=ai.getNumAttributes();
		int attMap[numAtt];

		for(i=0;i<numAtt;i++) attMap[i]=0;

		char string[1000];
		JVector<int>sizes;
		JVector<int *>BBs;
		FILE *fp=fopen("cutPoints.dat","r");
		if(fp==NULL) {
			fprintf(stderr,"Cannot open cutPoints.dat\n");
			exit(1);
		}
		fgets(string,999,fp);
		while(!feof(fp)) {
			char *token;
        		token=strtok(string," ");
			JVector<int> varsBB;
			while(token!=NULL) {
				int value=atoi(token);
				if(value<0 || value>=numAtt) {
					fprintf(stderr,"Attribute %d is out of ranges\n",value);
					exit(1);
				}
				if(attMap[value]==1) {
					fprintf(stderr,"Attribute %d has already been used\n",value);
					exit(1);
				}
				attMap[value]=1;
				varsBB.addElement(value);
        			token=strtok(NULL," ");
			}
	
			int size=varsBB.size();
			int *bb=new int[size];	
			for(i=0;i<size;i++) {
				bb[i]=varsBB[i];
			}
	
			sizes.addElement(size);
			BBs.addElement(bb);
			fgets(string,999,fp);
		}

		numBB=sizes.size();
		sizeBBs=new int[numBB];
		defBBs = new int *[numBB];
		for(i=0;i<numBB;i++) {
			sizeBBs[i]=sizes[i];
			defBBs[i]=BBs[i];
		}

		fclose(fp);

		for(i=0;i<numAtt;i++) {
			if(attMap[i]==0) {
				fprintf(stderr,"Attribute %d is not included in any BB\n",i);
				exit(1);
			}
		}
	}
		
}

void timerCrossover::newIteration(int iteration,int lastIteration)
{
}

void timerCrossover::dumpStats(int iteration)
{
}
