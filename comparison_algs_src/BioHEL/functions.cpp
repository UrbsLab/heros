#ifndef __CUDA_COMPILED__
#define __CUDA_COMPILED__
#endif

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <time.h>

#include "classifier.h"
#include "instance.h"
#include "agentPerformanceTraining.h"
#include "classifier_hyperrect_list.h"
#include "classifier_hyperrect_list_real.h"
#include "classifier_gabil.h"
#include "configManagement.h"

extern configManagement cm;

using namespace std;

struct ClassifierInfo {
    int numAtt;
    int predictedClass;
};

int deviceSelected = -1;
size_t memDevice = (size_t) 0;
size_t memPerBlock = (size_t) 0;
int tPerBlock;

int instancesPerRun;
int classifiersPerRun;
int maxNumAtt = 0;
int ruleSize = 0;
int atts = 0;

float * realValues;
int * realClasses;
//int * typeOfAttributes;
int realClassesSize;
int realValuesSize;
int typeOfAttSize;

unsigned char * chromosome;
float * predicates;
int * whichAtt;

ClassifierInfo * info;
int * offsetPred;

int predSize;
int whichSize;

int infoSize;
int offsetPredSize = 0;

int alreadyAllocatedIns = 0;

extern "C"int **calculateFitnessCudaMixed(int alreadyAllocatedInstances,
                                          int maxNumAtt, int ruleSize, int atts, int numInstances, int popSize,
                                          float *predicates, int predSize, int *whichAtt, int whichSize,
                                          ClassifierInfo * info, int infoSize, int * offsetPred,
                                          int offsetPredSize, float *realValues, int realValuesSize,
                                          int *realClasses, int realClassesSize, int * typeOfAttributes,
                                          int typeOfAttSize, int instancesPerRun, int classifiersPerRun,
                                          int strataOffset);

extern "C"int **calculateFitnessCudaReal(int alreadyAllocatedInstances,
                                         int maxNumAtt, int atts, int numInstances, int popSize,
                                         float *predicates, int predSize, int *whichAtt, int whichSize,
                                         ClassifierInfo * info, int infoSize, float *realValues,
                                         int realValuesSize, int *realClasses, int realClassesSize,
                                         int instancesPerRun, int classifiersPerRun, int strataOffset);

extern "C" int **calculateFitnessCudaNominal(int alreadyAllocatedInstances,
                                             int numInstances, int popSize, int atts, int ruleSize,
                                             unsigned char *chromosome, int chromosomeSize, float *realValues,
                                             int realValuesSize, int *realClasses, int realClassesSize,
                                             int instancesPerRun, int classifiersPerRun, int strataOffset);

extern "C" void setDevice(size_t * memDevice, size_t * memPerBlock, int * threadsPerBlock,
                          int * deviceSelected, double percent);

extern "C"void allocateInstanceMemoryCuda(int realValuesSize, int realClassesSize);

extern "C"void copyInstancesToDeviceCudaReal(int numInstances, int atts,
                                             int *instancesPerRun, float *realValues, int realValuesSize,
                                             int *realClasses, int realClassesSize);

extern "C"void copyInstancesToDeviceCudaMixed(int numInstances, int atts,
                                              int *instancesPerRun, float *realValues, int realValuesSize,
                                              int *realClasses, int realClassesSize, int * typeOfAttributes,
                                              int typeOfAttSize);

extern "C" void freeInstanceMemory();

extern "C" void copyStaticClassifierInfo(int atts, int *offsetPredicates,
                                         int offsetPredSize);

extern timerSymbolicKR *tSymbolic;

inline int getMemPerClassifierMixed() {
    return sizeof(ClassifierInfo) + sizeof(float) * ruleSize + sizeof(int)
            * maxNumAtt * 2;
}

inline int getMemPerClassifierReal() {
    return sizeof(ClassifierInfo) + sizeof(float) * maxNumAtt * 2 + sizeof(int)
            * maxNumAtt;
}

inline int getMemPerClassifierNominal() {
    return sizeof(unsigned char) * ruleSize + sizeof(int);
}

inline int getMemPerInstance() {
    return sizeof(int) * atts + sizeof(int);
}

inline int getAdditionalMem() {
    return 3 * sizeof(int);
}

inline int memoryFit(int instancesPerRun, int classifiersPerRun, int memPerInstance, int memPerClassifier, int additionalMem, size_t memDevice) {

    return memPerInstance*instancesPerRun + memPerClassifier*classifiersPerRun
            + additionalMem*ceil((double)instancesPerRun/tPerBlock)*classifiersPerRun + additionalMem*classifiersPerRun <= memDevice;
}

void doMemoryCalculation(int memPerClassifier, int numInstances, int popSize) {

    // Memory per instance. Realvalues + real class
    int memPerInstance = getMemPerInstance();

    // Stores information shared between each pair of instance and classifier
    int additionalMem = getAdditionalMem();

    // Calculate if we can fit all instances in memory when considering
    // working with only one classifier
    instancesPerRun = numInstances;

    if (memoryFit(instancesPerRun,1,memPerInstance,memPerClassifier,additionalMem,memDevice)) {
        // If all the instances fit in memory calculate how many classifiers fit in memory
        classifiersPerRun = popSize;

        int i=2;
        while(!memoryFit(instancesPerRun,classifiersPerRun,memPerInstance,memPerClassifier,additionalMem,memDevice) && classifiersPerRun > 0) {
            if (classifiersPerRun == 1) {
                classifiersPerRun = 0;
                break;
            }
            classifiersPerRun = min(classifiersPerRun-1,(int) ceil((double)popSize/i++));
        }

    } else {
        // If not all the instances fit in memory we calculate the number of classifier
        // we could work with considering only one instance.
        classifiersPerRun = popSize;
        if (memoryFit(1,classifiersPerRun,memPerInstance,memPerClassifier,additionalMem,memDevice)) {
            // If all the classifiers fit in memory we recalculate de number of instances.
            instancesPerRun = (int) ceil((double)numInstances/2);

            int i=3;
            while(!memoryFit(instancesPerRun,classifiersPerRun,memPerInstance,memPerClassifier,additionalMem,memDevice) && instancesPerRun > 0) {
                if (instancesPerRun == 1) {
                    instancesPerRun = 0;
                    break;
                }
                instancesPerRun = min(instancesPerRun-1,(int) ceil((double)numInstances/i++));
            }

        } else {
            // Dedicate half the memory for the classifiers and half for the instances.
            // This is a suboptimal solution. This should consider the shared amount of
            // data additionalMem. A Binary search should be done at this point

            double a = ((double) additionalMem*memPerClassifier)/((double) tPerBlock*memPerInstance);
            double b = (double) 2*memPerClassifier + 2*additionalMem;
            double c = (double) -1*memDevice;

            classifiersPerRun = (int) floor((-b + sqrt(pow(b,2) - 4*a*c))/((double)2*a));

            instancesPerRun = numInstances;

            int i=2;
            while(!memoryFit(instancesPerRun,classifiersPerRun,memPerInstance,memPerClassifier,additionalMem,memDevice) && instancesPerRun > 0) {

                if (instancesPerRun == 1) {
                    instancesPerRun = 0;
                    break;
                }
                instancesPerRun = min(instancesPerRun-1,(int) ceil((double)numInstances/i++));
            }
        }
    }

    // If at least one calculated value is 0 we abort the execution.
    if(instancesPerRun < 1 || classifiersPerRun < 1) {
        fprintf(stderr,"It is not possible to store these problem instances in device memory. Please use the serial version.\n");
        exit(1);
    }

    printf("Ins per run %d Class per run %d \n", instancesPerRun,classifiersPerRun);

}

void doMemoryCalculationInstances(int memPerClassifier, int numInstances) {

    // Memory per instance. Realvalues + real class
    int memPerInstance = getMemPerInstance();

    // Stores information shared between each pair of instance and classifier
    int additionalMem = getAdditionalMem();

    // We check if we can fit all the instances in memory
    instancesPerRun = numInstances;
    classifiersPerRun = 0;
    if (!memoryFit(instancesPerRun,1,memPerInstance,memPerClassifier,additionalMem,memDevice)) {
        instancesPerRun = 0;
    }

}


void flattenClassifiersReal(classifier ** population, int popSize) {
    // Generating the population structures
    predicates = (float *) malloc(sizeof(float) * popSize * maxNumAtt * 2);
    whichAtt = (int *) malloc(sizeof(int) * popSize * maxNumAtt);
    info = (ClassifierInfo *) malloc(sizeof(ClassifierInfo) * popSize);

    for (int i = 0; i < popSize; i++) {
        classifier_hyperrect_list_real * ind =
                static_cast<classifier_hyperrect_list_real*> (population[i]);

        info[i].numAtt = ind->numAtt;

        bcopy(ind->whichAtt, &whichAtt[i * maxNumAtt], sizeof(int)
              * info[i].numAtt);
        bcopy(ind->predicates, &predicates[i * maxNumAtt * 2], sizeof(float)
              * info[i].numAtt * 2);

        info[i].predictedClass = ind->classValue;


    }
}

void flattenClassifiersNominal(classifier ** population, int popSize) {
    // Generating the population structures
    chromosome = (unsigned char *) malloc(sizeof(int) * ruleSize * popSize);

    for (int i = 0; i < popSize; i++) {
        classifier_gabil * ind = static_cast<classifier_gabil*> (population[i]);

        bcopy(ind->chromosome, &chromosome[i * ruleSize], sizeof(unsigned char)
              * ruleSize);

    }
}

void flattenClassifiersMixed(classifier ** population, int popSize) {
    // Generating the population structures
    predicates = (float *) malloc(sizeof(float) * popSize * ruleSize);

    whichAtt = (int *) malloc(sizeof(int) * popSize * maxNumAtt);
    info = (ClassifierInfo *) malloc(sizeof(ClassifierInfo) * popSize);
    offsetPred = (int *) malloc(sizeof(int) * popSize * maxNumAtt);

    for (int i = 0; i < popSize; i++) {
        classifier_hyperrect_list * ind =
                static_cast<classifier_hyperrect_list *> (population[i]);

        info[i].numAtt = ind->numAtt;

        bcopy(ind->whichAtt, &whichAtt[i * maxNumAtt], sizeof(int)
              * info[i].numAtt);
        bcopy(ind->predicates, &predicates[i * ruleSize], sizeof(float)
              * ind->ruleSize);
        bcopy(ind->offsetPredicates, &offsetPred[i * maxNumAtt], sizeof(int)
              * info[i].numAtt);

        info[i].predictedClass = ind->classValue;


    }
}

void freeClassifiersReal() {

    free(predicates);
    free(info);
    free(whichAtt);

}

void freeClassifiersNominal() {

    free(chromosome);
}

void freeClassifiersMixed() {

    free(predicates);
    free(info);
    free(whichAtt);
    free(offsetPred);

}

void freeInstances() {
    free(realValues);
    free(realClasses);
}

//This function is only called from the main in case the instances were allocated at the beginning
extern "C"void freeAllInstanceMemory() {

    if(alreadyAllocatedIns) {
        freeInstances();
        freeInstanceMemory();
    }
}

void flattenInstances(instance ** instances, int numInstances) {

    // Generating the instances structures
    realValues = (float *) malloc(sizeof(float) * numInstances * atts);
    realClasses = (int *) malloc(sizeof(int) * numInstances);

    for (int i = 0; i < numInstances; i++) {

        bcopy(instances[i]->realValues, &realValues[i * atts], sizeof(float)
              * atts);
        realClasses[i] = instances[i]->instanceClass;
    }
}

inline void getMaxNumAttReal(classifier ** population, int popSize) {

    maxNumAtt = 0;
    for (int i = 0; i < popSize; i++) {
        classifier_hyperrect_list_real* ind =
                static_cast<classifier_hyperrect_list_real*> (population[i]);
        if (ind->numAtt > maxNumAtt) {
            maxNumAtt = ind->numAtt;
        }

    }
}

inline void getMaxNumAttMixed(classifier ** population, int popSize) {

    maxNumAtt = 0;
    ruleSize = 0;
    for (int i = 0; i < popSize; i++) {
        classifier_hyperrect_list* ind =
                static_cast<classifier_hyperrect_list*> (population[i]);
        if (ind->numAtt > maxNumAtt) {
            maxNumAtt = ind->numAtt;
        }

        if (ind->ruleSize > ruleSize) {
            ruleSize = ind->ruleSize;
        }

    }

}

extern "C"void setDeviceCuda() {

    double percent;
    if(cm.thereIsParameter(PERC_DEVICE_MEM)) {
       percent = cm.getParameter(PERC_DEVICE_MEM);
    } else {
       percent = 1;
    }
    if(cm.thereIsParameter(DEVICE_SELECTED)) {
        deviceSelected = (int) cm.getParameter(DEVICE_SELECTED);
    }

    setDevice(&memDevice, &memPerBlock, &tPerBlock, &deviceSelected, percent);

}

inline void setAtts() {
    atts = tGlobals->numAttributesMC;
    maxNumAtt = atts;
    ruleSize = tReal->ruleSize;
}

inline void setNumAttsReal(classifier ** population, int popSize) {
    getMaxNumAttReal(population, popSize);
}

inline void setNumAttsNominal(int myruleSize) {
    if (ruleSize == 0)
        ruleSize = myruleSize;
}

inline void setNumAttsMixed(classifier ** population, int popSize) {
    getMaxNumAttMixed(population, popSize);

}

inline void setInstanceSizes() {

    realClassesSize = sizeof(int) * instancesPerRun;
    realValuesSize = sizeof(float) * instancesPerRun * atts;
    typeOfAttSize = sizeof(int) * atts;

}

inline void setClassifierSizesReal() {

    predSize = sizeof(float) * classifiersPerRun * maxNumAtt * 2;
    whichSize = sizeof(int) * classifiersPerRun * maxNumAtt;
    if (infoSize == 0)
        infoSize = sizeof(ClassifierInfo) * classifiersPerRun;
}

inline void setOffsetPredSize() {
    offsetPredSize = sizeof(int) * (atts + 1);
}

inline void setClassifierSizesNominal() {
    predSize = sizeof(int) * ruleSize * classifiersPerRun;
}

inline void setClassifierSizesMixed() {

    predSize = sizeof(float) * classifiersPerRun * ruleSize;
    whichSize = sizeof(int) * classifiersPerRun * maxNumAtt;
    if (infoSize == 0)
        infoSize = sizeof(ClassifierInfo) * classifiersPerRun;
    offsetPredSize = sizeof(int) * classifiersPerRun * maxNumAtt;
}

extern "C" void copyInstancesToDeviceReal(instance ** instances, int numInstances) {
    setAtts();

    doMemoryCalculationInstances(getMemPerClassifierReal(), numInstances);

    if (instancesPerRun < numInstances) {
        alreadyAllocatedIns = 0;
        return;
    }

    setInstanceSizes();
    flattenInstances(instances, numInstances);

    allocateInstanceMemoryCuda(realValuesSize, realClassesSize);
    copyInstancesToDeviceCudaReal(numInstances, atts, &instancesPerRun,
                                  realValues, realValuesSize, realClasses, realClassesSize);

    alreadyAllocatedIns = 1;
}

extern "C" void copyInstancesToDeviceNominal(instance ** instances, int numInstances) {

    setAtts();

    setOffsetPredSize();

    copyStaticClassifierInfo(atts, tSymbolic->offsetAttribute,
                             offsetPredSize);

    doMemoryCalculationInstances(getMemPerClassifierNominal(), numInstances);

    if (instancesPerRun < numInstances) {
        alreadyAllocatedIns = 0;
        return;
    }

    setInstanceSizes();
    flattenInstances(instances, numInstances);

    allocateInstanceMemoryCuda(realValuesSize, realClassesSize);
    copyInstancesToDeviceCudaReal(numInstances, atts, &instancesPerRun,
                                  realValues, realValuesSize, realClasses, realClassesSize);

    alreadyAllocatedIns = 1;

}

extern "C" void copyInstancesToDeviceMixed(instance ** instances, int * typeOfAttributes,
                                           int numInstances) {


    setAtts();

    doMemoryCalculationInstances(getMemPerClassifierMixed(), numInstances);

    if (instancesPerRun < numInstances) {
        alreadyAllocatedIns = 0;
        return;
    }

    setInstanceSizes();
    flattenInstances(instances, numInstances);

    allocateInstanceMemoryCuda(realValuesSize, realClassesSize);
    copyInstancesToDeviceCudaMixed(numInstances, atts, &instancesPerRun,
                                   realValues, realValuesSize, realClasses, realClassesSize,
                                   typeOfAttributes, typeOfAttSize);

    alreadyAllocatedIns = 1;


}

extern "C"void calculateFitnessReal(classifier ** population,
                                    instance ** instances, int popSize, int numInstances, int strataOffset) {

    //Classifiers information
    setNumAttsReal(population, popSize);

    doMemoryCalculation(getMemPerClassifierReal(), numInstances, popSize);
    setInstanceSizes();

    if (!alreadyAllocatedIns) {
        strataOffset = 0;
        flattenInstances(instances, numInstances);
    }

    setClassifierSizesReal();
    flattenClassifiersReal(population, popSize);


    int **counters = calculateFitnessCudaReal(alreadyAllocatedIns, maxNumAtt,
                                              atts, numInstances, popSize, predicates, predSize, whichAtt,
                                              whichSize, info, infoSize, realValues, realValuesSize, realClasses,
                                              realClassesSize, instancesPerRun, classifiersPerRun, strataOffset);

    for (int i = 0; i < popSize; i++) {
        classifier_hyperrect_list_real* ind =
		static_cast<classifier_hyperrect_list_real*> (population[i]);
        agentPerformanceTraining * aps = new agentPerformanceTraining(
                numInstances, ind->getClass());
        aps->setNumMatched(counters[i][0]);
        aps->setNumPos(counters[i][1]);
        aps->setNumOK(counters[i][2]);

        ind->setAccuracy(aps->getAccuracy());
        ind->setAccuracy2(aps->getAccuracy2());
        ind->setCoverage(aps->getCoverage());

        ind->modif = 0;
        ind->fitness = aps->getFitness(*ind);
    }

    if (!alreadyAllocatedIns)
        freeInstances();
    freeClassifiersReal();
    free(counters);

}

extern "C" void calculateFitnessNominal(classifier ** population,
                                        instance ** instances, int popSize, int numInstances,
                                        int strataOffset) {

    //Classifiers information
    setNumAttsNominal(tSymbolic->ruleSize);

    doMemoryCalculation(getMemPerClassifierNominal(), numInstances, popSize);
    setInstanceSizes();

    if (!alreadyAllocatedIns) {
        strataOffset = 0;
        flattenInstances(instances, numInstances);
    }

    setClassifierSizesNominal();
    flattenClassifiersNominal(population, popSize);

    int** counters = calculateFitnessCudaNominal(alreadyAllocatedIns,
                                                 numInstances, popSize, atts, tSymbolic->ruleSize, chromosome, predSize,
                                                 realValues, realValuesSize, realClasses, realClassesSize,
                                                 instancesPerRun, classifiersPerRun, strataOffset);

    for (int i = 0; i < popSize; i++) {
        classifier_gabil* ind = static_cast<classifier_gabil*> (population[i]);
        agentPerformanceTraining * aps = new agentPerformanceTraining(
                numInstances, ind->getClass());
        aps->setNumMatched(counters[i][0]);
        aps->setNumPos(counters[i][1]);
        aps->setNumOK(counters[i][2]);

        ind->setAccuracy(aps->getAccuracy());
        ind->setAccuracy2(aps->getAccuracy2());
        ind->setCoverage(aps->getCoverage());

        ind->modif = 0;
        ind->fitness = aps->getFitness(*ind);
    }

    if (!alreadyAllocatedIns)
        freeInstances();
    freeClassifiersNominal();
    free(counters);

}

extern "C" void calculateFitnessMixed(classifier ** population,
                                      instance ** instances, int * typeOfAttributes, int popSize,
                                      int numInstances, int strataOffset) {

    //Classifiers information
    for (int i = 0; i < popSize; i++) {
        classifier_hyperrect_list* ind =
		static_cast<classifier_hyperrect_list*> (population[i]);
        ind->initiateEval();
    }

    setNumAttsMixed(population, popSize);

    doMemoryCalculation(getMemPerClassifierMixed(), numInstances, popSize);

    setInstanceSizes();

    if (!alreadyAllocatedIns) {
        strataOffset = 0;
        flattenInstances(instances, numInstances);
    }

    setClassifierSizesMixed();
    flattenClassifiersMixed(population, popSize);

    int** counters = calculateFitnessCudaMixed(alreadyAllocatedIns, maxNumAtt,
                                               ruleSize, atts, numInstances, popSize, predicates, predSize,
                                               whichAtt, whichSize, info, infoSize, offsetPred, offsetPredSize,
                                               realValues, realValuesSize, realClasses, realClassesSize,
                                               typeOfAttributes, typeOfAttSize, instancesPerRun,
                                               classifiersPerRun, strataOffset);

    for (int i = 0; i < popSize; i++) {
        classifier_hyperrect_list * ind =
		static_cast<classifier_hyperrect_list *> (population[i]);
        agentPerformanceTraining * aps = new agentPerformanceTraining(
                numInstances, ind->getClass());

        //printf("Matched %d Pos %d OK %d\n",counters[i][0],counters[i][1],counters[i][2]);
        aps->setNumMatched(counters[i][0]);
        aps->setNumPos(counters[i][1]);
        aps->setNumOK(counters[i][2]);

        ind->setAccuracy(aps->getAccuracy());
        ind->setAccuracy2(aps->getAccuracy2());
        ind->setCoverage(aps->getCoverage());

        ind->modif = 0;
        ind->fitness = aps->getFitness(*ind);

        ind->finalizeEval();
    }

    if (!alreadyAllocatedIns)
        freeInstances();
    freeClassifiersMixed();
    free(counters);

}
