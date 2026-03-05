#ifndef _MATCH_PROFILE_AGENT_
#define _MATCH_PROFILE_AGENT_

#include "JVector.h"

class matchProfileAgent
{
public:
    unsigned long long numInstances;
    unsigned long long numMatched;
    unsigned char *mapOK;
    unsigned char *mapKO;
    unsigned long long *listOK;
    unsigned long long *listKO;
    unsigned long long numOK;
    unsigned long long numKO;
    int ruleClass;

    matchProfileAgent(unsigned long long numInstances,int pRuleClass);
    ~matchProfileAgent();
    inline void addOK(unsigned long long instance)
    {
        listOK[numOK++]=instance;
    }
    inline void addKO(unsigned long long instance)
    {
        listKO[numKO++]=instance;
    }
    void generateProfiles();
    void removeMatched(unsigned long long *instances,unsigned long long numInstances);
};


#endif
