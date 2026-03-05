#include <math.h>
#include <cstdio>
#include <cstring>
#include <iostream>

#include "classifier.h"
#include "classifier_aggregated.h"
#include "attributesInfo.h"
#include "timerGlobals.h"
#include "timerMDL.h"
#include "agentPerformance.h"
#include "agentPerformanceTraining.h"
#include "instanceSet.h"
#include "JVector.h"
#include "messageBuffer.h"
#include "factory.h"

extern attributesInfo ai;
extern instanceSet *is;
extern timerGlobals *tGlobals;
extern timerMDL *tMDL;
extern int lastIteration;
extern messageBuffer mb;
extern int nodeRank;

using namespace std;

void classifierStats(classifier_aggregated & ind, instanceSet * is,
                     const char *typeOfFile) {
    int i;
    agentPerformance ap(ind.getNumClassifiers(), ai.getNumClasses());
    //instanceSet *is = new instanceSet(instancesFile, TEST);
    int numInstances = is->getNumInstancesOfIteration();

    for (i = 0; i < numInstances; i++) {
        instance *ins = is->getInstance(i);
        int realClass = ins->getClass();
        int predictedClass = -1;
        int whichClassifier = ind.classify(ins);
        if (whichClassifier != -1) {
            predictedClass = ind.getClass(whichClassifier);
        }
        ap.addPrediction(realClass, predictedClass, whichClassifier);
    }

    ind.setAccuracy(ap.getAccuracy());
    ap.dumpStats(typeOfFile);
    //delete is;
}

int isMajority(classifier & ind) {
    int i;
    int numInstances = is->getNumInstances();
    instance **instances = is->getAllInstances();

    int cl = ind.getClass();

    int nc = ai.getNumClasses();
    int classCounts[nc];
    for (i = 0; i < nc; i++)
        classCounts[i] = 0;

    ind.initiateEval();

    int numPos = 0;
    for (i = 0; i < numInstances; i++) {
        if (instances[i]->instanceClass == cl)
            numPos++;
        if (ind.doMatch(instances[i])) {
            classCounts[instances[i]->instanceClass]++;
        }
    }

    ind.finalizeEval();

    double ratio = (double) classCounts[cl] / (double) numPos;
    if (ratio < tMDL->coverageBreaks[cl] / 3)
        return 0;

    int max = classCounts[0];
    int posMax = 0;
    int tie = 0;

    for (i = 1; i < nc; i++) {

        if (classCounts[i] > max) {
            max = classCounts[i];
            posMax = i;
            tie = 0;
        } else if (classCounts[i] == max) {
            tie = 1;
        }
    }

    return (max > 0 && !tie && posMax == cl);
}

void classifierBriefTest(classifier_aggregated & ind, instanceSet *is) {
    int i;
    agentPerformance ap(ind.getNumClassifiers(), ai.getNumClasses());
    int numInstances = is->getNumInstancesOrig();
    instance **instances = is->getOrigInstances();

    for (i = 0; i < numInstances; i++) {
        int predictedClass = -1;
        int whichClassifier = ind.classify(instances[i]);
        if (whichClassifier != -1) {
            predictedClass = ind.getClass(whichClassifier);
        }
        ap.addPrediction(instances[i]->instanceClass, predictedClass,
                         whichClassifier);
    }

    ap.dumpStatsBrief();
}

double classifierFitness(classifier & ind) {
    int i;

    int cl = ind.getClass();
    int numInstances = is->getNumInstancesOfIteration();
    agentPerformanceTraining ap(numInstances, cl);
    instance **instances = is->getInstancesOfIteration();

    ind.initiateEval();

    for (i = 0; i < numInstances; i++) {
        if (ind.doMatch(instances[i])) {
            ap.addMatch(instances[i]->instanceClass, cl);
        } else {
            ap.addNoMatch(instances[i]->instanceClass);
        }
    }
    ind.finalizeEval();

    ind.setAccuracy(ap.getAccuracy());
    ind.setAccuracy2(ap.getAccuracy2());
    ind.setCoverage(ap.getCoverage());

    ind.setNumInstancesMatched(ap.getNumMatched());
    ind.setNumInstancesPosOK(ap.getNumOK());
    ind.setNumInstancesPos(ap.getNumPos());

    double fitness = ap.getFitness(ind);
    return fitness;
}

int ** classifierFitnessWithMatchSet(classifier & ind) {
    int i;
    int numInstances = is->getNumInstancesOfIteration();
    int cl = ind.getClass();
    agentPerformanceTraining ap(numInstances, cl);
    instance **instances = is->getInstancesOfIteration();

    JVector<int> pos;

    ind.initiateEval();

    for (i = 0; i < numInstances; i++) {
        if (ind.doMatch(instances[i])) {
            ap.addMatch(instances[i]->instanceClass, cl);
            if (instances[i]->instanceClass == cl)
                pos.addElement(i);

        } else {
            ap.addNoMatch(instances[i]->instanceClass);
        }
    }
    ind.finalizeEval();

    ind.setAccuracy(ap.getAccuracy());
    ind.setAccuracy2(ap.getAccuracy2());
    ind.setCoverage(ap.getCoverage());

    //double fitness=ap.getFitness(ind);

    int possize = pos.size();
    int ** res = new int *[2];
    res[0] = new int[possize];
    res[1] = new int[1];

    for (i = 0; i < possize; i++) {
        res[0][i] = pos[i];
    }

    res[1][0] = possize;

    return res;
    //pos.clear();
    //neg.clear();
    //return matched;
}

int ** classifierFitnessWithMatchSetComplete(classifier & ind) {
    int i;
    int numInstances = is->getNumInstancesOfIteration();
    int cl = ind.getClass();
    agentPerformanceTraining ap(numInstances, cl);
    instance **instances = is->getInstancesOfIteration();

    JVector<int> pos;
    JVector<int> neg;

    ind.initiateEval();

    for (i = 0; i < numInstances; i++) {
        if (ind.doMatch(instances[i])) {
            ap.addMatch(instances[i]->instanceClass, cl);
            if (instances[i]->instanceClass == cl) {
                pos.addElement(i);
            } else {
                neg.addElement(i);
            }

        } else {
            ap.addNoMatch(instances[i]->instanceClass);

        }
    }
    ind.finalizeEval();

    ind.setAccuracy(ap.getAccuracy());
    ind.setAccuracy2(ap.getAccuracy2());
    ind.setCoverage(ap.getCoverage());

    //double fitness=ap.getFitness(ind);

    int possize = pos.size();
    int negsize = neg.size();
    int ** res = new int *[3];
    res[0] = new int[possize];
    //res[1] = new int[1];
    res[1] = new int[negsize];
    res[2] = new int[2];


    for (i = 0; i < possize; i++) {
        res[0][i] = pos[i];
    }

    for (i = 0; i < negsize; i++) {
        res[1][i] = neg[i];
    }



    res[2][0] = possize;
    res[2][1] = negsize;

    return res;
}

agentPerformance calculateAgentPerformance(classifier_aggregated & ind) {

    int i;
    agentPerformance ap(ind.getNumClassifiers(), ai.getNumClasses());
    int numInstances = is->getNumInstancesOfIteration();

    for (i = 0; i < numInstances; i++) {
        instance *ins = is->getInstance(i);
        int realClass = ins->getClass();
        int predictedClass = -1;

        int whichClassifier = ind.classify(ins);

        if (whichClassifier != -1) {
            predictedClass = ind.getClass(whichClassifier);
        }
        ap.addPrediction(realClass, predictedClass, whichClassifier);
    }

    return ap;

}



