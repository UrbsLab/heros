#include "matchProfileAgent.h"

matchProfileAgent::matchProfileAgent(unsigned long long pNumInstances,int pRuleClass)
{
    numInstances=pNumInstances;
    ruleClass=pRuleClass;

    listOK=new unsigned long long[numInstances];
    listKO=new unsigned long long[numInstances];
    numOK=numKO=0;
}

matchProfileAgent::~matchProfileAgent()
{
    delete mapOK;
    delete mapKO;
    delete listOK;
    delete listKO;
}

void matchProfileAgent::generateProfiles()
{
    unsigned long long i;

    mapOK=new unsigned char[numInstances];
    mapKO=new unsigned char[numInstances];

    bzero(mapOK,numInstances*sizeof(unsigned char));
    bzero(mapKO,numInstances*sizeof(unsigned char));

    for(i=0; i<numOK; i++)
    {
        mapOK[listOK[i]]=1;
    }

    for(i=0; i<numKO; i++)
    {
        mapKO[listKO[i]]=1;
    }

    numMatched=numOK+numKO;
}


void matchProfileAgent::removeMatched(unsigned long long *instances,unsigned long long numInst)
{
    unsigned long long i;
    unsigned long long instOK[numInst];
    unsigned long long instKO[numInst];
    unsigned long long removedOK=0;
    unsigned long long removedKO=0;

    for(i=0; i<numInst; i++)
    {
        unsigned long long inst=instances[i];
        if(mapOK[inst])
        {
            mapOK[inst]=0;
            instOK[removedOK++]=inst;
        }
        else
        {
            mapKO[inst]=0;
            instKO[removedKO++]=inst;
        }
    }

    if(removedOK)
    {
        unsigned long long index=0;
        unsigned long long numRemoved=1;
        while(listOK[index]<instOK[0]) index++;
        while(numRemoved<removedOK)
        {
            while(listOK[index+numRemoved]<instOK[numRemoved])
            {
                listOK[index]=listOK[index+numRemoved];
                index++;
            }
            numRemoved++;
        }
        while(index+numRemoved<numOK)
        {
            listOK[index]=listOK[index+numRemoved];
            index++;
        }
        numOK-=removedOK;
    }

    if(removedKO)
    {
        unsigned long long index=0;
        unsigned long long numRemoved=1;
        while(listKO[index]<instKO[0]) index++;
        while(numRemoved<removedKO)
        {
            while(listKO[index+numRemoved]<instKO[numRemoved])
            {
                listKO[index]=listKO[index+numRemoved];
                index++;
            }

            numRemoved++;
        }
        while(index+numRemoved<numKO)
        {
            listKO[index]=listKO[index+numRemoved];
            index++;
        }
        numKO-=removedKO;
    }

    numMatched=numOK+numKO;
}
