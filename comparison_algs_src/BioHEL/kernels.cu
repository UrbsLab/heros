#ifndef __CUDA_COMPILED__
#define __CUDA_COMPILED__
#endif

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <cutil_inline.h>

using namespace std;

//#define THREADS_PER_BLOCK 512
#define MAX_TYPE_SIZE 512
//#define N 512
#define NOMINAL 1
#define ENTER 2
#define REAL 3

#ifdef __DEVICE_EMULATION__
#define EMUSYNC __syncthreads()
#else
#define EMUSYNC
#endif

struct __align__ (8) ClassifierInfo {
    int numAtt;
    int predictedClass;
};

unsigned char *d_chromosome;
float *d_predicates;
int *d_whichAtt;
ClassifierInfo *d_info;
int *d_offsetPred;

float *d_realValues;
int *d_realClasses;
__constant__ int c_typeOfAttribute[MAX_TYPE_SIZE];
__constant__ int c_numAtts[1];
__constant__ int c_offsetAttribute[MAX_TYPE_SIZE];
int threadsPerBlock;


//template < unsigned int blockSize >
__global__ void reduction6(int *entrada, int * last, int totalObjects,
                                   int arraySize, int a);
//template < unsigned int blockSize >
__global__ static void cudaCalculateMatchMixed(int insPerRun,
                                                       int classPerRun,
                                                       int maxNumAtt, int ruleSize, int numAttIns,
                                                       float *predicates,
                                                       int *whichAtt,
                                                       ClassifierInfo * info,
                                                       int * offsetPredicates,
                                                       float *realValues,
                                                       int *realClasses,
                                                       int *numIns,
                                                       int *finalStruct);

//template < unsigned int blockSize >
__global__ static void cudaCalculateMatchReal(int insPerRun,
                                                      int classPerRun,
                                                      int maxNumAtt, int numAttIns,
                                                      float *predicates,
                                                      int *whichAtt,
                                                      ClassifierInfo * info,
                                                      float *realValues,
                                                      int *realClasses,
                                                      int *numIns,
                                                      int *finalStruct);

//template < unsigned int blockSize >
__global__ static void cudaCalculateMatchNominal(int insPerRun,
                                                         int classPerRun,
                                                         int ruleSize,
                                                         unsigned char *chromosome,
                                                         float *realValues,
                                                         int *realClasses,
                                                         int *numIns);

inline int setDeviceFirstTime() {

    // Determine de number of cuda enabled devices available.
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    int deviceSelected;

    // If there is a device available we iterate over the device to select the device with the larger capacity
    if (deviceCount > 0) {
        int device;

        int maxCapacityFound = 0;

        cudaDeviceProp prop;
        for (device = 0; device < deviceCount; device++) {

            cudaGetDeviceProperties(&prop, device);

            if (prop.totalGlobalMem > maxCapacityFound) {
                maxCapacityFound = prop.totalGlobalMem;
                deviceSelected = device;
                // A device with a larger capacity was found
            }

        }

        return deviceSelected;
    } else {
        fprintf(stderr, "There are not CUDA enabled devices available");
        exit(1);
    }
}

extern "C" void setDevice(size_t * memDevice, size_t * memPerBlock, int * tBlock,
                          int * deviceSelected, double percent) {


    // Sets 256MB as the global memory in case of using device emulation.
#ifdef __DEVICE_EMULATION__

    *memDevice = 268435456;
    *memPerBlock = 16384;
    *tBlock = 512
    threadsPerBlock = 512;
#endif

#ifndef __DEVICE_EMULATION__
    cudaDeviceProp prop;
    if (*deviceSelected == -1) {
        *deviceSelected = setDeviceFirstTime();
    }

    if (*memDevice == 0 || *memPerBlock == 0 && *tBlock == 0) {
        // If a device was already set previously we collect the data
        cudaSetDevice(*deviceSelected);
        cudaGetDeviceProperties(&prop, *deviceSelected);
//        if(*deviceSelected == 0) {
            //double percent = cm.getParameter(PERC_DEVICE_MEM);
            *memDevice = (size_t) floor(percent*prop.totalGlobalMem);
            fprintf(stdout, "Using %f of device memory %lld\n", percent, prop.totalGlobalMem);
//        } else {
//            *memDevice = prop.totalGlobalMem;
//        }
        *memPerBlock = prop.sharedMemPerBlock;
        *tBlock = prop.maxThreadsPerBlock;
        threadsPerBlock = prop.maxThreadsPerBlock;
        //threadsPerBlock = 512;

       // *memDevice = 1024*12;

        fprintf(stdout, "Set mem device %lld memBlock %lld threads %d Device %d\n", *memDevice, *memPerBlock, threadsPerBlock, *deviceSelected);

    }
#endif

}

void launchMatchReal(int instancesPerRun, int classifiersPerRun, int blockX,
                     int maxNumAtt, int numAttIns, int shared_mem_size, int *d_numIns,
                     int *finalStruct,
                     int strataOffset) {

    // Initialize block and grid size
    dim3 grid = dim3(blockX, classifiersPerRun, 1);
    dim3 threads = dim3(threadsPerBlock, 1, 1);


    // Calculate the match and predicted for each instance and classifier pair.
    cudaCalculateMatchReal<<< grid, threads,
    shared_mem_size * 3 >>> (instancesPerRun, classifiersPerRun,
                             maxNumAtt, numAttIns, d_predicates,
                             d_whichAtt, d_info, &d_realValues[strataOffset*numAttIns],
                             &d_realClasses[strataOffset], d_numIns, finalStruct);
    cutilCheckMsg("Kernel execution failed");
    cudaThreadSynchronize();


}

void launchMatchNominal(int instancesPerRun, int classifiersPerRun, int blockX,
                        int shared_mem_size, int atts, int ruleSize, int *d_numIns,
                        int strataOffset) {

    // Initialize block and grid size
    dim3 grid = dim3(blockX, classifiersPerRun, 1);
    dim3 threads = dim3(threadsPerBlock, 1, 1);


    // Calculate the match and predicted for each instance and classifier pair.
    cudaCalculateMatchNominal<<< grid, threads,
    shared_mem_size * 3 >>> (instancesPerRun, classifiersPerRun, ruleSize,
                             d_chromosome,
                             &d_realValues[strataOffset*atts],
                             &d_realClasses[strataOffset], d_numIns);
    cutilCheckMsg("Kernel execution failed");
    cudaThreadSynchronize();

}

void launchMatchMixed(int instancesPerRun, int classifiersPerRun, int blockX,
                      int maxNumAtt, int ruleSize, int numAttIns, int shared_mem_size,
                      int *d_numIns, int *finalStruct, int strataOffset) {


    // Initialize block and grid size
    dim3 grid = dim3(blockX, classifiersPerRun, 1);
    dim3 threads = dim3(threadsPerBlock, 1, 1);

    // Calculate the match and predicted for each instance and classifier pair.
    cudaCalculateMatchMixed<<< grid, threads,
    shared_mem_size * 3 >>> (instancesPerRun, classifiersPerRun,
                             maxNumAtt, ruleSize, numAttIns, d_predicates,
                             d_whichAtt, d_info, d_offsetPred, &d_realValues[strataOffset*numAttIns],
                             &d_realClasses[strataOffset], d_numIns, finalStruct);
    cutilCheckMsg("Kernel execution failed");
    cudaThreadSynchronize();

}

void launchReduction(int insPerRun, int classPerRun, int shared_mem_size,
                     int *d_numIns, int *finalStruct) {


    int blockX =
            (int) ceil((double) insPerRun / ((double) threadsPerBlock * 2));

    //Iterates over the three areas created in the first kernel
    for (unsigned int a = 0; a < 3; a++) {

        int offset = a * insPerRun * classPerRun;
        unsigned int numThreads = insPerRun;



//        int numBlocks = (int) ceil(blockX / (double) N);
        int numBlocks = blockX;

        //Runs the reduction until the number of blocks is 0
        while (numBlocks > 0) {
            // setup execution parameters
            dim3 grid(numBlocks, classPerRun, 1);
            dim3 threads(threadsPerBlock, 1, 1);

            //OJO
            reduction6<<< grid, threads,
            shared_mem_size >>> (&d_numIns[offset], finalStruct,
                                 numThreads, insPerRun,a);
            cudaThreadSynchronize();
            cutilCheckMsg("Kernel execution failed");

            numThreads = numBlocks;
            numBlocks = (numBlocks == 1 ? 0 : (int) ceil((double) numThreads
                                                         / (double) (threadsPerBlock)));
//            numBlocks = (numBlocks == 1 ? 0 : (int) ceil((double) numThreads
//                                                         / (double) (threadsPerBlock * N)));

        }

    }

}

inline void launchKernelsReal(int instancesPerRun, int classifiersPerRun,
                              int maxNumAtt, int numAttIns, int classChecked, int *d_numIns, int *finalStruct,
                              int **counters, int strataOffset) {



    unsigned int shared_mem_size = sizeof(int) * threadsPerBlock;

    int blockX = (int) ceil((double) instancesPerRun
                            /
                            (double) threadsPerBlock);

    // ************ first kernel **********

    launchMatchReal(instancesPerRun, classifiersPerRun, blockX, maxNumAtt,
                    numAttIns, shared_mem_size, d_numIns, finalStruct, strataOffset);

    // *********** second kernel **********

    if (blockX > 1) {
        launchReduction(blockX, classifiersPerRun, shared_mem_size, d_numIns, finalStruct);
    }


    // Copy the memory from device to host and the organize it into de counter structure
    int size = sizeof(int) * classifiersPerRun * 3;
    int * result = (int *) malloc(size);
    cutilSafeCall(cudaMemcpy(result, finalStruct, size, cudaMemcpyDeviceToHost));

    for (unsigned int classi = 0; classi < classifiersPerRun; classi++) {

        int offset = classi * 3;
        counters[classChecked + classi][0] += result[offset];
        counters[classChecked + classi][1] += result[offset + 1];
        counters[classChecked + classi][2] += result[offset + 2];

    }


}

inline void launchKernelsNominal(int instancesPerRun, int classifiersPerRun,
                                 int classChecked, int atts, int ruleSize, int *d_numIns, int **counters,
                                 int strataOffset) {

    unsigned int shared_mem_size = sizeof(int) * threadsPerBlock;

    int blockX = (int) ceil((double) instancesPerRun
                            /
                            (double) threadsPerBlock);

    // ************ first kernel **********

    launchMatchNominal(instancesPerRun, classifiersPerRun, blockX,
                       shared_mem_size, atts, ruleSize, d_numIns, strataOffset);

    // *********** second kernel **********

    if (blockX > 1) {
        launchReduction(blockX, classifiersPerRun, shared_mem_size, d_numIns, d_numIns);
    }

    int size = sizeof(int) * classifiersPerRun * 3;
    int * result = (int *) malloc(size);
    cutilSafeCall(cudaMemcpy(result, d_numIns, size, cudaMemcpyDeviceToHost));

    for (unsigned int classi = 0; classi < classifiersPerRun; classi++) {

        int offset = classi * 3;
        counters[classChecked + classi][0] += result[offset];
        counters[classChecked + classi][1] += result[offset + 1];
        counters[classChecked + classi][2] += result[offset + 2];

    }


}

inline void launchKernelsMixed(int instancesPerRun, int classifiersPerRun,
                               int maxNumAtt, int ruleSize, int numAttIns, int classChecked,
                               int *d_numIns, int *finalStruct, int **counters, int strataOffset) {


    unsigned int shared_mem_size = sizeof(int) * threadsPerBlock;

    int blockX = (int) ceil((double) instancesPerRun
                            /
                            (double) threadsPerBlock);

    // ************ first kernel **********

    launchMatchMixed(instancesPerRun, classifiersPerRun, blockX, maxNumAtt,
                     ruleSize, numAttIns, shared_mem_size, d_numIns, finalStruct, strataOffset);

    // *********** second kernel **********

    if (blockX > 1) {
        launchReduction(blockX, classifiersPerRun, shared_mem_size, d_numIns, finalStruct);
    }

    // Copy the memory from device to host and the organize it into de counter structure
    int size = sizeof(int) * classifiersPerRun * 3;
    int * result = (int *) malloc(size);
    cutilSafeCall(cudaMemcpy(result, finalStruct, size, cudaMemcpyDeviceToHost));

    for (unsigned int classi = 0; classi < classifiersPerRun; classi++) {

        int offset = classi * 3;
        counters[classChecked + classi][0] += result[offset];
        counters[classChecked + classi][1] += result[offset + 1];
        counters[classChecked + classi][2] += result[offset + 2];

    }

}

inline void allocateInstanceMemory(int realValuesSize, int realClassesSize) {
    // Allocating instance memory
    cutilSafeCall(cudaMalloc((void **) &d_realValues, realValuesSize));
    cutilSafeCall(cudaMalloc((void **) &d_realClasses, realClassesSize));

}

extern "C" void freeInstanceMemory() {

    // Setting free the instance memory
    cudaFree(d_realValues);
    cudaFree(d_realClasses);

}

inline void allocateClassifiersMemoryReal(int predSize, int whichSize,
                                          int infoSize) {
    // Allocating real classifiers memory
    cutilSafeCall(cudaMalloc((void **) &d_whichAtt, whichSize));
    cutilSafeCall(cudaMalloc((void **) &d_predicates, predSize));
    cutilSafeCall(cudaMalloc((void **) &d_info, infoSize));
}

inline void allocateClassifiersMemoryNominal(int ruleSize) {
    cutilSafeCall(cudaMalloc((void **) &d_chromosome, ruleSize));

}

inline void allocateClassifiersMemoryMixed(int predSize, int whichSize,
                                           int infoSize, int offsetPredSize) {
    // Allocating mixed classifiers memory
    cutilSafeCall(cudaMalloc((void **) &d_whichAtt, whichSize));
    cutilSafeCall(cudaMalloc((void **) &d_predicates, predSize));
    cutilSafeCall(cudaMalloc((void **) &d_info, infoSize));
    cutilSafeCall(cudaMalloc((void **) &d_offsetPred, offsetPredSize));
}

inline void freeClassifiersMemoryReal() {
    // Seting free the classifier memory
    cudaFree(d_predicates);
    cudaFree(d_info);
    cudaFree(d_whichAtt);

}

inline void freeClassifiersMemoryNominal() {
    // Seting free the classifier memory
    cudaFree(d_chromosome);

}

inline void freeClassifiersMemoryMixed() {
    // Seting free the classifier memory
    cudaFree(d_predicates);
    cudaFree(d_info);
    cudaFree(d_whichAtt);
    cudaFree(d_offsetPred);

}


extern "C" void copyStaticClassifierInfo(int atts, int *offsetPredicates,
                                         int offsetPredSize) {

    // This information is static a doesn't not chance during the whole execution of the GA.
    cutilSafeCall(cudaMemcpyToSymbol(c_numAtts, &atts, 1, 0,
                                     cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMemcpyToSymbol(c_offsetAttribute, offsetPredicates,
                                     offsetPredSize, 0, cudaMemcpyHostToDevice));

}

inline void copyInstanceMemoryReal(int numInstances, int insChecked, int atts,
                                   int *instancesPerRun, float *realValues, int realValuesSize,
                                   int *realClasses, int realClassesSize) {

    // Adjusting the instances per run parameter for the last iteration
    if (*instancesPerRun > numInstances - insChecked) {
        *instancesPerRun = numInstances - insChecked;

        realClassesSize = sizeof(int) * (*instancesPerRun);
        realValuesSize = sizeof(float) * (*instancesPerRun) * atts;

    }


    // Copying the instance data into the device memory
    cutilSafeCall(cudaMemcpy(d_realValues, &(realValues[insChecked * atts]),
                             realValuesSize, cudaMemcpyHostToDevice));

    cutilSafeCall(cudaMemcpy(d_realClasses, &(realClasses[insChecked]),
                             realClassesSize, cudaMemcpyHostToDevice));


}

inline void copyInstanceMemoryMixed(int numInstances, int insChecked, int atts,
                                    int *instancesPerRun, float *realValues, int realValuesSize,
                                    int *realClasses, int realClassesSize, int * typeOfAttributes,
                                    int typeOfAttSize) {


    // Adjusting the instances per run parameter for the last iteration
    if (*instancesPerRun > numInstances - insChecked) {
        *instancesPerRun = numInstances - insChecked;

        realClassesSize = sizeof(int) * (*instancesPerRun);
        realValuesSize = sizeof(float) * (*instancesPerRun) * atts;

    }


    // Copying the instance data into the device memory
    cutilSafeCall(cudaMemcpy(d_realValues, &(realValues[insChecked * atts]),
                             realValuesSize, cudaMemcpyHostToDevice));

    cutilSafeCall(cudaMemcpy(d_realClasses, &(realClasses[insChecked]),
                             realClassesSize, cudaMemcpyHostToDevice));

    cutilSafeCall(cudaMemcpyToSymbol(c_typeOfAttribute, typeOfAttributes,
                                     typeOfAttSize, 0, cudaMemcpyHostToDevice));


}


// This function is called by functions.cpp to allocate the instances at the beginning
extern "C" void allocateInstanceMemoryCuda(int realValuesSize, int realClassesSize) {
    allocateInstanceMemory(realValuesSize, realClassesSize);
}

// This function is called by functions.cpp to copy the instances at the beginning
extern "C" void copyInstancesToDeviceCudaReal(int numInstances, int atts,
                                              int *instancesPerRun, float *realValues, int realValuesSize,
                                              int *realClasses, int realClassesSize) {

    copyInstanceMemoryReal(numInstances, 0, atts, instancesPerRun, realValues,
                           realValuesSize, realClasses, realClassesSize);

}

// This function is called by functions.cpp to copy the instances at the beginning
extern "C" void copyInstancesToDeviceCudaMixed(int numInstances, int atts,
                                               int *instancesPerRun, float *realValues, int realValuesSize,
                                               int *realClasses, int realClassesSize, int * typeOfAttributes,
                                               int typeOfAttSize) {


    copyInstanceMemoryMixed(numInstances, 0, atts, instancesPerRun, realValues,
                            realValuesSize, realClasses, realClassesSize, typeOfAttributes,
                            typeOfAttSize);

}



inline void copyClassifiersMemoryReal(int popSize, int classChecked,
                                      int maxNumAtt, int *classifiersPerRun, float *predicates, int predSize,
                                      int *whichAtt, int whichSize, ClassifierInfo * info, int infoSize) {


    // Adjusting the classifiers per run for the last iterations
    if (*classifiersPerRun > popSize - classChecked) {
        *classifiersPerRun = popSize - classChecked;

        predSize = sizeof(float) * (*classifiersPerRun) * maxNumAtt * 2;
        whichSize = sizeof(int) * (*classifiersPerRun) * maxNumAtt;

    }

    // Copying pop info into the device memory
    cutilSafeCall(cudaMemcpy(d_predicates, &(predicates[classChecked
                                                        * maxNumAtt * 2]), predSize, cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMemcpy(d_whichAtt, &(whichAtt[classChecked * maxNumAtt]),
                             whichSize, cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMemcpy(d_info, &(info[classChecked]), infoSize,
                             cudaMemcpyHostToDevice));


}

inline void copyClassifiersMemoryNominal(int popSize, int classChecked,
                                         int *classifiersPerRun, int ruleSize, unsigned char *chromosome,
                                         int chromosomeSize) {


    // Adjusting the classifiers per run for the last iterations
    if (*classifiersPerRun > popSize - classChecked) {
        *classifiersPerRun = popSize - classChecked;

        chromosomeSize = sizeof(unsigned char) * (*classifiersPerRun)
                         * ruleSize;

    }

    // Copying pop info into the device memory
    cutilSafeCall(cudaMemcpy(d_chromosome,
                             &(chromosome[classChecked * ruleSize]), chromosomeSize,
                             cudaMemcpyHostToDevice));


}

inline void copyClassifiersMemoryMixed(int popSize, int classChecked,
                                       int maxNumAtt, int ruleSize, int *classifiersPerRun, float *predicates,
                                       int predSize, int *whichAtt, int whichSize, ClassifierInfo * info,
                                       int infoSize, int * offsetPred, int offsetPredSize) {


    // Adjusting the classifiers per run for the last iterations
    if (*classifiersPerRun > popSize - classChecked) {
        *classifiersPerRun = popSize - classChecked;

        predSize = sizeof(float) * (*classifiersPerRun) * ruleSize;
        whichSize = sizeof(int) * (*classifiersPerRun) * maxNumAtt;

        offsetPredSize = whichSize;

    }

    // Copying pop info into the device memory
    cutilSafeCall(cudaMemcpy(d_predicates,
                             &(predicates[classChecked * ruleSize]), predSize,
                             cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMemcpy(d_whichAtt, &(whichAtt[classChecked * maxNumAtt]),
                             whichSize, cudaMemcpyHostToDevice));

    cutilSafeCall(cudaMemcpy(d_info, &(info[classChecked]), infoSize,
                             cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMemcpy(d_offsetPred, &(offsetPred[classChecked
                                                        * maxNumAtt]), offsetPredSize, cudaMemcpyHostToDevice));


}

inline void iteratingOverClassifiersReal(int popSize, int numInstances,
                                         int maxNumAtt, int atts, int classifiersPerRun, int instancesPerRun,
                                         float *predicates, int predSize, int *whichAtt, int whichSize,
                                         ClassifierInfo * info, int infoSize, float *realValues,
                                         int realValuesSize, int *realClasses, int realClassesSize,
                                         int *d_numIns, int *finalStruct, int **counters, int strataOffset) {

    // Iterating over the classifiers  to copy the info into device memory and calculate de counters
    int classifiersPerRunOrig = classifiersPerRun;
    for (int classChecked = 0; classChecked < popSize; classChecked
         = classChecked + classifiersPerRun) {

        copyClassifiersMemoryReal(popSize, classChecked, maxNumAtt,
                                  &classifiersPerRun, predicates, predSize, whichAtt, whichSize,
                                  info, infoSize);

        int instancesPerRunOrig = instancesPerRun;
        // Iterate over the instances to copy into device memory and calculate the counters
        for (int insChecked = 0; insChecked < numInstances; insChecked
             = insChecked + instancesPerRun) {

            copyInstanceMemoryReal(numInstances, insChecked, atts,
                                   &instancesPerRun, realValues, realValuesSize, realClasses,
                                   realClassesSize);

            launchKernelsReal(instancesPerRun, classifiersPerRun, maxNumAtt,
                              atts, classChecked, d_numIns, finalStruct, counters, strataOffset);

        }

        instancesPerRun = instancesPerRunOrig;

    }
    classifiersPerRun = classifiersPerRunOrig;
}

inline void iteratingOverClassifiersNominal(int popSize, int numInstances,
                                            int atts, int ruleSize, int classifiersPerRun, int instancesPerRun,
                                            unsigned char * chromosome, int chromosomeSize, float *realValues,
                                            int realValuesSize, int *realClasses, int realClassesSize,
                                            int *d_numIns, int **counters, int strataOffset) {

    // Iterating over the classifiers  to copy the info into device memory and calculate de counters
    int classifiersPerRunOrig = classifiersPerRun;
    for (int classChecked = 0; classChecked < popSize; classChecked
         = classChecked + classifiersPerRun) {

        copyClassifiersMemoryNominal(popSize, classChecked, &classifiersPerRun,
                                     ruleSize, chromosome, chromosomeSize);

        int instancesPerRunOrig = instancesPerRun;
        // Iterate over the instances to copy into device memory and calculate the counters
        for (int insChecked = 0; insChecked < numInstances; insChecked
             = insChecked + instancesPerRun) {

            copyInstanceMemoryReal(numInstances, insChecked, atts,
                                   &instancesPerRun, realValues, realValuesSize, realClasses,
                                   realClassesSize);

            launchKernelsNominal(instancesPerRun, classifiersPerRun,
                                 classChecked, atts, ruleSize, d_numIns, counters, strataOffset);

        }

        instancesPerRun = instancesPerRunOrig;

    }
    classifiersPerRun = classifiersPerRunOrig;
}

inline void iteratingOverClassifiersMixed(int popSize, int numInstances,
                                          int maxNumAtt, int ruleSize, int atts, int classifiersPerRun,
                                          int instancesPerRun, float *predicates, int predSize, int *whichAtt,
                                          int whichSize, ClassifierInfo * info, int infoSize, int * offsetPred,
                                          int offsetPredSize, float *realValues, int realValuesSize,
                                          int *realClasses, int realClassesSize, int * typeOfAttributes,
                                          int typeOfAttSize, int *d_numIns, int *finalStruct, int **counters, int strataOffset) {

    // Iterating over the classifiers  to copy the info into device memory and calculate de counters
    int classifiersPerRunOrig = classifiersPerRun;
    for (int classChecked = 0; classChecked < popSize; classChecked
         = classChecked + classifiersPerRun) {

        copyClassifiersMemoryMixed(popSize, classChecked, maxNumAtt, ruleSize,
                                   &classifiersPerRun, predicates, predSize, whichAtt, whichSize,
                                   info, infoSize, offsetPred, offsetPredSize);

        int instancesPerRunOrig = instancesPerRun;
        // Iterate over the instances to copy into device memory and calculate the counters
        for (int insChecked = 0; insChecked < numInstances; insChecked
             = insChecked + instancesPerRun) {

            copyInstanceMemoryMixed(numInstances, insChecked, atts,
                                    &instancesPerRun, realValues, realValuesSize, realClasses,
                                    realClassesSize, typeOfAttributes, typeOfAttSize);

            launchKernelsMixed(instancesPerRun, classifiersPerRun, maxNumAtt,
                               ruleSize, atts, classChecked, d_numIns, finalStruct, counters, strataOffset);

        }

        instancesPerRun = instancesPerRunOrig;

    }
    classifiersPerRun = classifiersPerRunOrig;
}

inline void iteratingOverInstancesReal(int popSize, int numInstances,
                                       int maxNumAtt, int atts, int classifiersPerRun, int instancesPerRun,
                                       float *predicates, int predSize, int *whichAtt, int whichSize,
                                       ClassifierInfo * info, int infoSize, float *realValues,
                                       int realValuesSize, int *realClasses, int realClassesSize,
                                       int *d_numIns, int *finalStruct, int **counters, int strataOffset) {

    // Iterate over the instances to copy into device memory and calculate the counters
    int instancesPerRunOrig = instancesPerRun;
    for (int insChecked = 0; insChecked < numInstances; insChecked
         += instancesPerRun) {

        copyInstanceMemoryReal(numInstances, insChecked, atts,
                               &instancesPerRun, realValues, realValuesSize, realClasses,
                               realClassesSize);

        int classifiersPerRunOrig = classifiersPerRun;
        // Iterating over the classifiers  to copy the info into device memory and calculate de counters
        for (int classChecked = 0; classChecked < popSize; classChecked
             += classifiersPerRun) {

            copyClassifiersMemoryReal(popSize, classChecked, maxNumAtt,
                                      &classifiersPerRun, predicates, predSize, whichAtt,
                                      whichSize, info, infoSize);

            launchKernelsReal(instancesPerRun, classifiersPerRun, maxNumAtt,
                              atts, classChecked, d_numIns,finalStruct, counters, strataOffset);

        }

        classifiersPerRun = classifiersPerRunOrig;

    }
    instancesPerRun = instancesPerRunOrig;
}

inline void iteratingOverInstancesNominal(int popSize, int numInstances,
                                          int atts, int ruleSize, int classifiersPerRun, int instancesPerRun,
                                          unsigned char * chromosome, int chromosomeSize, float *realValues,
                                          int realValuesSize, int *realClasses, int realClassesSize,
                                          int *d_numIns, int **counters, int strataOffset) {

    // Iterate over the instances to copy into device memory and calculate the counters
    int instancesPerRunOrig = instancesPerRun;
    for (int insChecked = 0; insChecked < numInstances; insChecked
         += instancesPerRun) {

        copyInstanceMemoryReal(numInstances, insChecked, atts,
                               &instancesPerRun, realValues, realValuesSize, realClasses,
                               realClassesSize);

        int classifiersPerRunOrig = classifiersPerRun;
        // Iterating over the classifiers  to copy the info into device memory and calculate de counters
        for (int classChecked = 0; classChecked < popSize; classChecked
             += classifiersPerRun) {

            copyClassifiersMemoryNominal(popSize, classChecked,
                                         &classifiersPerRun, ruleSize, chromosome, chromosomeSize);

            launchKernelsNominal(instancesPerRun, classifiersPerRun,
                                 classChecked, atts, ruleSize, d_numIns, counters, strataOffset);

        }

        classifiersPerRun = classifiersPerRunOrig;

    }
    instancesPerRun = instancesPerRunOrig;
}

inline void iteratingOverInstancesMixed(int popSize, int numInstances,
                                        int maxNumAtt, int ruleSize, int atts, int classifiersPerRun,
                                        int instancesPerRun, float *predicates, int predSize, int *whichAtt,
                                        int whichSize, ClassifierInfo * info, int infoSize, int * offsetPred,
                                        int offsetPredSize, float *realValues, int realValuesSize,
                                        int *realClasses, int realClassesSize, int * typeOfAttributes,
                                        int typeOfAttSize, int *d_numIns, int *finalStruct, int **counters, int strataOffset) {

    // Iterate over the instances to copy into device memory and calculate the counters
    int instancesPerRunOrig = instancesPerRun;
    for (int insChecked = 0; insChecked < numInstances; insChecked
         += instancesPerRun) {

        copyInstanceMemoryMixed(numInstances, insChecked, atts,
				&instancesPerRun, realValues, realValuesSize, realClasses,
				realClassesSize, typeOfAttributes, typeOfAttSize);


        int classifiersPerRunOrig = classifiersPerRun;
        // Iterating over the classifiers  to copy the info into device memory and calculate de counters
        for (int classChecked = 0; classChecked < popSize; classChecked
             += classifiersPerRun) {

            copyClassifiersMemoryMixed(popSize, classChecked, maxNumAtt,
                                       ruleSize, &classifiersPerRun, predicates, predSize,
                                       whichAtt, whichSize, info, infoSize, offsetPred,
                                       offsetPredSize);

            launchKernelsMixed(instancesPerRun, classifiersPerRun, maxNumAtt,
                               ruleSize, atts, classChecked, d_numIns, finalStruct, counters, strataOffset);

        }

        classifiersPerRun = classifiersPerRunOrig;

    }
    instancesPerRun = instancesPerRunOrig;
}

void onlyIterateClassifiersReal(int popSize, int maxNumAtt, int atts,
                                int classifiersPerRun, int instancesPerRun, float *predicates,
                                int predSize, int *whichAtt, int whichSize, ClassifierInfo * info,
                                int infoSize, int *d_numIns, int *finalStruct, int **counters, int strataOffset) {

    for (int classChecked = 0; classChecked < popSize; classChecked
         += classifiersPerRun) {

        copyClassifiersMemoryReal(popSize, classChecked, maxNumAtt,
                                  &classifiersPerRun, predicates, predSize, whichAtt, whichSize,
                                  info, infoSize);

        launchKernelsReal(instancesPerRun, classifiersPerRun, maxNumAtt, atts,
                          classChecked, d_numIns, finalStruct, counters, strataOffset);

    }
}

void onlyIterateClassifiersNominal(int popSize, int classifiersPerRun,
                                   int instancesPerRun, int atts, int ruleSize, unsigned char *chromosome,
                                   int chromosomeSize, int *d_numIns, int **counters, int strataOffset) {

    for (int classChecked = 0; classChecked < popSize; classChecked
         += classifiersPerRun) {

        copyClassifiersMemoryNominal(popSize, classChecked, &classifiersPerRun,
                                     ruleSize, chromosome, chromosomeSize);

        launchKernelsNominal(instancesPerRun, classifiersPerRun, classChecked,
                             atts, ruleSize, d_numIns, counters, strataOffset);

    }
}

void onlyIterateClassifiersMixed(int popSize, int maxNumAtt, int ruleSize,
                                 int atts, int classifiersPerRun, int instancesPerRun,
                                 float *predicates, int predSize, int *whichAtt, int whichSize,
                                 ClassifierInfo * info, int infoSize, int * offsetPred,
                                 int offsetPredSize, int *d_numIns, int *finalStruct, int **counters, int strataOffset) {

    for (int classChecked = 0; classChecked < popSize; classChecked
         += classifiersPerRun) {

        copyClassifiersMemoryMixed(popSize, classChecked, maxNumAtt, ruleSize,
                                   &classifiersPerRun, predicates, predSize, whichAtt, whichSize,
                                   info, infoSize, offsetPred, offsetPredSize);

        launchKernelsMixed(instancesPerRun, classifiersPerRun, maxNumAtt,
                           ruleSize, atts, classChecked, d_numIns, finalStruct, counters, strataOffset);

    }
}

extern "C" int **calculateFitnessCudaReal(int alreadyAllocatedInstances,
                                          int maxNumAtt, int atts, int numInstances, int popSize,
                                          float *predicates, int predSize, int *whichAtt, int whichSize,
                                          ClassifierInfo * info, int infoSize, float *realValues,
                                          int realValuesSize, int *realClasses, int realClassesSize,
                                          int instancesPerRun, int classifiersPerRun, int strataOffset) {

    // Initializing the counters for each classifier. This counters will be updated
    // after each run, because it is possible that we wont be able to check all the
    // classifiers at the same time.
    int **counters = (int **) malloc(sizeof(int *) * popSize);
    for (int i = 0; i < popSize; i++) {
        counters[i] = (int *) malloc(sizeof(int) * 3);
        counters[i][0] = 0;
        counters[i][1] = 0;
        counters[i][2] = 0;
    }

    // Reserving device memory for instances
    if (!alreadyAllocatedInstances)
        allocateInstanceMemory(realValuesSize, realClassesSize);

    //Reserving device memory for classifiers
    allocateClassifiersMemoryReal(predSize, whichSize, infoSize);



    // Initialize the device output memory
    int *d_numIns;
    int numInsSize = sizeof(int) * 3 * classifiersPerRun * (int) ceil(
            (double) instancesPerRun / (double) threadsPerBlock);

    cutilSafeCall(cudaMalloc((void **) &d_numIns, numInsSize));

    int *finalStruct;
    cutilSafeCall(cudaMalloc((void **) &finalStruct, sizeof(int) * 3 * classifiersPerRun));

    if (alreadyAllocatedInstances) {
        onlyIterateClassifiersReal(popSize, maxNumAtt, atts, classifiersPerRun,
                                   instancesPerRun, predicates, predSize, whichAtt, whichSize,
                                   info, infoSize, d_numIns, finalStruct, counters, strataOffset);

    } else if (classifiersPerRun == popSize) {
        iteratingOverClassifiersReal(popSize, numInstances, maxNumAtt, atts,
                                     classifiersPerRun, instancesPerRun, predicates, predSize,
                                     whichAtt, whichSize, info, infoSize, realValues,
                                     realValuesSize, realClasses, realClassesSize, d_numIns, finalStruct,
                                     counters,strataOffset);
    } else {
        iteratingOverInstancesReal(popSize, numInstances, maxNumAtt, atts,
                                   classifiersPerRun, instancesPerRun, predicates, predSize,
                                   whichAtt, whichSize, info, infoSize, realValues,
                                   realValuesSize, realClasses, realClassesSize, d_numIns, finalStruct,
                                   counters,strataOffset);
    }

    if (!alreadyAllocatedInstances)
        freeInstanceMemory();
    freeClassifiersMemoryReal();
    cudaFree(d_numIns);
    cudaFree(finalStruct);

    return counters;

}

extern "C" int **calculateFitnessCudaNominal(int alreadyAllocatedInstances,
                                             int numInstances, int popSize, int atts, int ruleSize,
                                             unsigned char *chromosome, int chromosomeSize, float *realValues,
                                             int realValuesSize, int *realClasses, int realClassesSize,
                                             int instancesPerRun, int classifiersPerRun, int strataOffset) {

    // Initializing the counters for each classifier. This counters will be updated
    // after each run, because it is possible that we wont be able to check all the
    // classifiers at the same time.
    int **counters = (int **) malloc(sizeof(int *) * popSize);
    for (int i = 0; i < popSize; i++) {
        counters[i] = (int *) malloc(sizeof(int) * 3);
        counters[i][0] = 0;
        counters[i][1] = 0;
        counters[i][2] = 0;
    }

    // Reserving device memory for instances
    if (!alreadyAllocatedInstances)
        allocateInstanceMemory(realValuesSize, realClassesSize);

    //Reserving device memory for classifiers
    allocateClassifiersMemoryNominal(ruleSize);

    // Initialize the device output memory
    int *d_numIns;
    int numInsSize = sizeof(int) * 3 * classifiersPerRun * (int) ceil(
            (double) instancesPerRun / (double) threadsPerBlock);
    cutilSafeCall(cudaMalloc((void **) &d_numIns, numInsSize));

    if (alreadyAllocatedInstances) {
        onlyIterateClassifiersNominal(popSize, classifiersPerRun,
                                      instancesPerRun, atts, ruleSize, chromosome, chromosomeSize,
                                      d_numIns, counters, strataOffset);

    } else if (classifiersPerRun == popSize) {
        iteratingOverClassifiersNominal(popSize, numInstances, atts, ruleSize,
                                        classifiersPerRun, instancesPerRun, chromosome, chromosomeSize,
                                        realValues, realValuesSize, realClasses, realClassesSize,
                                        d_numIns, counters,strataOffset);
    } else {
        iteratingOverInstancesNominal(popSize, numInstances, atts, ruleSize,
                                      classifiersPerRun, instancesPerRun, chromosome, chromosomeSize,
                                      realValues, realValuesSize, realClasses, realClassesSize,
                                      d_numIns, counters,strataOffset);
    }

    if (!alreadyAllocatedInstances)
        freeInstanceMemory();
    freeClassifiersMemoryNominal();
    cudaFree(d_numIns);

    return counters;

}

extern "C" int **calculateFitnessCudaMixed(int alreadyAllocatedInstances,
                                           int maxNumAtt, int ruleSize, int atts, int numInstances, int popSize,
                                           float *predicates, int predSize, int *whichAtt, int whichSize,
                                           ClassifierInfo * info, int infoSize, int * offsetPred,
                                           int offsetPredSize, float *realValues, int realValuesSize,
                                           int *realClasses, int realClassesSize, int * typeOfAttributes,
                                           int typeOfAttSize, int instancesPerRun, int classifiersPerRun,
                                           int strataOffset) {

    // Initializing the counters for each classifier. This counters will be updated
    // after each run, because it is possible that we wont be able to check all the
    // classifiers at the same time.
    int **counters = (int **) malloc(sizeof(int *) * popSize);
    for (int i = 0; i < popSize; i++) {
        counters[i] = (int *) malloc(sizeof(int) * 3);
        counters[i][0] = 0;
        counters[i][1] = 0;
        counters[i][2] = 0;
    }

    // Reserving device memory for instances
    if (!alreadyAllocatedInstances) {
        allocateInstanceMemory(realValuesSize, realClassesSize);
    }

    //Reserving device memory for classifiers
    allocateClassifiersMemoryMixed(predSize, whichSize, infoSize,
                                   offsetPredSize);



    // Initialize the device output memory
    int *d_numIns;
    int numInsSize = sizeof(int) * 3 * classifiersPerRun * (int) ceil(
            (double) instancesPerRun / (double) threadsPerBlock);


    cutilSafeCall(cudaMalloc((void **) &d_numIns, numInsSize));

    int *finalStruct;
    cutilSafeCall(cudaMalloc((void **) &finalStruct, sizeof(int) * 3 * classifiersPerRun));

    if (alreadyAllocatedInstances) {
        onlyIterateClassifiersMixed(popSize, maxNumAtt, ruleSize, atts,
                                    classifiersPerRun, instancesPerRun, predicates, predSize,
                                    whichAtt, whichSize, info, infoSize, offsetPred,
                                    offsetPredSize, d_numIns, finalStruct, counters, strataOffset);

    } else if (classifiersPerRun == popSize) {
        iteratingOverClassifiersMixed(popSize, numInstances, maxNumAtt,
                                      ruleSize, atts, classifiersPerRun, instancesPerRun, predicates,
                                      predSize, whichAtt, whichSize, info, infoSize, offsetPred,
                                      offsetPredSize, realValues, realValuesSize, realClasses,
                                      realClassesSize, typeOfAttributes, typeOfAttSize, d_numIns, finalStruct,
                                      counters, strataOffset);
    } else {
        iteratingOverInstancesMixed(popSize, numInstances, maxNumAtt, ruleSize,
                                    atts, classifiersPerRun, instancesPerRun, predicates, predSize,
                                    whichAtt, whichSize, info, infoSize, offsetPred,
                                    offsetPredSize, realValues, realValuesSize, realClasses,
                                    realClassesSize, typeOfAttributes, typeOfAttSize, d_numIns, finalStruct,
                                    counters, strataOffset);
    }

    if (!alreadyAllocatedInstances)
        freeInstanceMemory();
    freeClassifiersMemoryMixed();
    cudaFree(d_numIns);
    cudaFree(finalStruct);

    return counters;

}

//template < unsigned int blockSize >
__global__ static void cudaCalculateMatchReal(int insPerRun,
                                                      int classPerRun,
                                                      int maxNumAtt, int numAttIns,
                                                      float *predicates,
                                                      int *whichAtt,
                                                      ClassifierInfo * info,
                                                      float *realValues,
                                                      int *realClasses,
                                                      int *numIns,
                                                      int *finalStruct)
{

    // Calculating the classifier and instance indexes inside the device structures
    int insIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int classIndex = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.x;

    extern __shared__ int sdata[];

    unsigned int tidDim = tid * 3;

    // If this data indexes exist
    if (insIndex < insPerRun && classIndex < classPerRun) {

        // Calculate match for the classifier and instance pair
	int attIndex = classIndex * maxNumAtt;
	int end=attIndex+info[classIndex].numAtt;
	int predOffset = attIndex * 2;
	int base = insIndex * numAttIns;
        int res = 1;

	for (; res && attIndex<end; attIndex++,predOffset+=2) {
            float value = realValues[base + whichAtt[attIndex]];
            if (value < predicates[predOffset]) res = 0;
            if (value > predicates[predOffset + 1]) res = 0;
        }

        int action = (realClasses[insIndex] == info[classIndex].predictedClass);

        sdata[tidDim] = res;
        sdata[tidDim + 1] = action;
        sdata[tidDim + 2] = action && res;
    } else {
        sdata[tidDim] = 0;
        sdata[tidDim + 1] = 0;
        sdata[tidDim + 2] = 0;
    }

    __syncthreads();

    // do reduction in shared mem
    if (blockDim.x == 1024 && tid < 512) {
        sdata[tidDim] += sdata[tidDim + 1536];
        sdata[tidDim + 1] += sdata[tidDim + 1537];
        sdata[tidDim + 2] += sdata[tidDim + 1538];
    }
    __syncthreads();
    if (tid < 256) {
        sdata[tidDim] += sdata[tidDim + 768];
        sdata[tidDim + 1] += sdata[tidDim + 769];
        sdata[tidDim + 2] += sdata[tidDim + 770];
    }
    __syncthreads();
    if (tid < 128) {
        sdata[tidDim] += sdata[tidDim + 384];
        sdata[tidDim + 1] += sdata[tidDim + 385];
        sdata[tidDim + 2] += sdata[tidDim + 386];
    }
    __syncthreads();
    if (tid < 64) {
        sdata[tidDim] += sdata[tidDim + 192];
        sdata[tidDim + 1] += sdata[tidDim + 193];
        sdata[tidDim + 2] += sdata[tidDim + 194];
    }
    __syncthreads();

#ifndef __DEVICE_EMULATION__
    if (tid < 32)
#endif
    {
	volatile int *sd=sdata;
	
        sd[tidDim] += sd[tidDim + 96];
        sd[tidDim + 1] += sd[tidDim + 97];
        sd[tidDim + 2] += sd[tidDim + 98];
        EMUSYNC;

        sd[tidDim] += sd[tidDim + 48];
        sd[tidDim + 1] += sd[tidDim + 49];
        sd[tidDim + 2] += sd[tidDim + 50];
        EMUSYNC;

        sd[tidDim] += sd[tidDim + 24];
        sd[tidDim + 1] += sd[tidDim + 25];
        sd[tidDim + 2] += sd[tidDim + 26];
        EMUSYNC;

        sd[tidDim] += sd[tidDim + 12];
        sd[tidDim + 1] += sd[tidDim + 13];
        sd[tidDim + 2] += sd[tidDim + 14];
        EMUSYNC;

        sd[tidDim] += sd[tidDim + 6];
        sd[tidDim + 1] += sd[tidDim + 7];
        sd[tidDim + 2] += sd[tidDim + 8];
        EMUSYNC;

        sd[tidDim] += sd[tidDim + 3];
        sd[tidDim + 1] += sd[tidDim + 4];
        sd[tidDim + 2] += sd[tidDim + 5];
        EMUSYNC;
    }

    if (tid == 0) {

        if (gridDim.x == 1) {

            int offset = classIndex*3;
            finalStruct[offset] = sdata[0];
            finalStruct[offset + 1] = sdata[1];
            finalStruct[offset + 2] = sdata[2];
        } else {

            int numInsIndex = classIndex * gridDim.x + blockIdx.x;
            int numInsOffset = gridDim.x * classPerRun;

            numIns[numInsIndex] = sdata[0];
            numInsIndex+=numInsOffset;
            numIns[numInsIndex] = sdata[1];
            numInsIndex+=numInsOffset;
            numIns[numInsIndex] = sdata[2];
        }

    }

}

//template < unsigned int blockSize >
__global__ static void cudaCalculateMatchNominal(int insPerRun,
                                                         int classPerRun,
                                                         int ruleSize,
                                                         unsigned char *chromosome,
                                                         float *realValues,
                                                         int *realClasses,
                                                         int *numIns)
{

    // Calculating the classifier and instance indexes inside the device structures
    int insIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int classIndex = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.x;

    extern __shared__ int sdata[];

    unsigned int tidDim = tid * 3;

    // If this data indexes exist
    if (insIndex < insPerRun && classIndex < classPerRun) {

        // Calculate match for the classifier and instance pair
        int j;
        int res = 1;


        for (j = 0; res && j < c_numAtts[0]; j++) {

            if (chromosome[classIndex * ruleSize + c_offsetAttribute[j]
                           + (unsigned char)realValues[insIndex * c_numAtts[0] + j]] == 0) {
                res = 0;
            }

        }

        int action =
		(realClasses[insIndex] ==
                 chromosome[classIndex*ruleSize + ruleSize - 1]);

        sdata[tidDim] = res;
        sdata[tidDim + 1] = action;
        sdata[tidDim + 2] = action && res;
    } else {
        sdata[tidDim] = 0;
        sdata[tidDim + 1] = 0;
        sdata[tidDim + 2] = 0;
    }

    __syncthreads();

    // do reduction in shared mem
    if (blockDim.x == 1024 && tid < 512) {
        sdata[tidDim] += sdata[tidDim + 1536];
        sdata[tidDim + 1] += sdata[tidDim + 1537];
        sdata[tidDim + 2] += sdata[tidDim + 1538];
    }
    __syncthreads();
    if (tid < 256) {
        sdata[tidDim] += sdata[tidDim + 768];
        sdata[tidDim + 1] += sdata[tidDim + 769];
        sdata[tidDim + 2] += sdata[tidDim + 770];
    }
    __syncthreads();
    if (tid < 128) {
        sdata[tidDim] += sdata[tidDim + 384];
        sdata[tidDim + 1] += sdata[tidDim + 385];
        sdata[tidDim + 2] += sdata[tidDim + 386];
    }
    __syncthreads();
    if (tid < 64) {
        sdata[tidDim] += sdata[tidDim + 192];
        sdata[tidDim + 1] += sdata[tidDim + 193];
        sdata[tidDim + 2] += sdata[tidDim + 194];
    }
    __syncthreads();

#ifndef __DEVICE_EMULATION__
    if (tid < 32)
#endif
    {
        sdata[tidDim] += sdata[tidDim + 96];
        sdata[tidDim + 1] += sdata[tidDim + 97];
        sdata[tidDim + 2] += sdata[tidDim + 98];
        EMUSYNC;

        sdata[tidDim] += sdata[tidDim + 48];
        sdata[tidDim + 1] += sdata[tidDim + 49];
        sdata[tidDim + 2] += sdata[tidDim + 50];
        EMUSYNC;

        sdata[tidDim] += sdata[tidDim + 24];
        sdata[tidDim + 1] += sdata[tidDim + 25];
        sdata[tidDim + 2] += sdata[tidDim + 26];
        EMUSYNC;

        sdata[tidDim] += sdata[tidDim + 12];
        sdata[tidDim + 1] += sdata[tidDim + 13];
        sdata[tidDim + 2] += sdata[tidDim + 14];
        EMUSYNC;

        sdata[tidDim] += sdata[tidDim + 6];
        sdata[tidDim + 1] += sdata[tidDim + 7];
        sdata[tidDim + 2] += sdata[tidDim + 8];
        EMUSYNC;

        sdata[tidDim] += sdata[tidDim + 3];
        sdata[tidDim + 1] += sdata[tidDim + 4];
        sdata[tidDim + 2] += sdata[tidDim + 5];
        EMUSYNC;
    }

    if (tid == 0) {

        if (gridDim.x == 1) {

            int offset = classIndex*3;
            numIns[offset] = sdata[0];
            numIns[offset + 1] = sdata[1];
            numIns[offset + 2] = sdata[2];
        } else {

            int numInsIndex = classIndex * gridDim.x + blockIdx.x;
            int numInsOffset = gridDim.x * classPerRun;

            numIns[numInsIndex] = sdata[0];
            numInsIndex+=numInsOffset;
            numIns[numInsIndex] = sdata[1];
            numInsIndex+=numInsOffset;
            numIns[numInsIndex] = sdata[2];
        }

    }

}

//template < unsigned int blockSize >
__global__ static void cudaCalculateMatchMixed(int insPerRun,
                                                       int classPerRun,
                                                       int maxNumAtt, int ruleSize, int numAttIns,
                                                       float *predicates,
                                                       int *whichAtt,
                                                       ClassifierInfo * info,
                                                       int * offsetPredicates,
                                                       float *realValues,
                                                       int *realClasses,
                                                       int *numIns,
                                                       int *finalStruct)
{

    // Calculating the classifier and instance indexes inside the device structures
    int insIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int classIndex = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.x;

    extern __shared__ int sdata[];

    unsigned int tidDim = tid * 3;

    // If this data indexes exist
    if (insIndex < insPerRun && classIndex < classPerRun) {

        // Calculate match for the classifier and instance pair
        int res = 1;
        int attIndex = classIndex * maxNumAtt;
        int end = attIndex+info[classIndex].numAtt;
        int baseI = insIndex * numAttIns;
        int baseR = classIndex * ruleSize;

        for (; res && attIndex<end; attIndex++) {
            int predOffset = baseR + offsetPredicates[attIndex];
            int att=whichAtt[attIndex];

            if(c_typeOfAttribute[att] == REAL) {
                float value = realValues[baseI + att];
                if (value < predicates[predOffset]) res = 0;
                if (value > predicates[predOffset + 1]) res = 0;
            } else {
                if(predicates[predOffset+(int)realValues[baseI + att]]==0) res = 0;
            }
        }


        int action = (realClasses[insIndex] == info[classIndex].predictedClass);

        sdata[tidDim] = res;
        sdata[tidDim + 1] = action;
        sdata[tidDim + 2] = action && res;
    } else {
        sdata[tidDim] = 0;
        sdata[tidDim + 1] = 0;
        sdata[tidDim + 2] = 0;
    }

    __syncthreads();

    // do reduction in shared mem
    if (blockDim.x == 1024 && tid < 512) {
        sdata[tidDim] += sdata[tidDim + 1536];
        sdata[tidDim + 1] += sdata[tidDim + 1537];
        sdata[tidDim + 2] += sdata[tidDim + 1538];
    }
    __syncthreads();
    if (tid < 256) {
        sdata[tidDim] += sdata[tidDim + 768];
        sdata[tidDim + 1] += sdata[tidDim + 769];
        sdata[tidDim + 2] += sdata[tidDim + 770];
    }
    __syncthreads();
    if (tid < 128) {
        sdata[tidDim] += sdata[tidDim + 384];
        sdata[tidDim + 1] += sdata[tidDim + 385];
        sdata[tidDim + 2] += sdata[tidDim + 386];
    }
    __syncthreads();
    if (tid < 64) {
        sdata[tidDim] += sdata[tidDim + 192];
        sdata[tidDim + 1] += sdata[tidDim + 193];
        sdata[tidDim + 2] += sdata[tidDim + 194];
    }
    __syncthreads();

#ifndef __DEVICE_EMULATION__
    if (tid < 32)
#endif
    {
	volatile int *sd=sdata;

        sd[tidDim] += sd[tidDim + 96];
        sd[tidDim + 1] += sd[tidDim + 97];
        sd[tidDim + 2] += sd[tidDim + 98];
        EMUSYNC;

        sd[tidDim] += sd[tidDim + 48];
        sd[tidDim + 1] += sd[tidDim + 49];
        sd[tidDim + 2] += sd[tidDim + 50];
        EMUSYNC;

        sd[tidDim] += sd[tidDim + 24];
        sd[tidDim + 1] += sd[tidDim + 25];
        sd[tidDim + 2] += sd[tidDim + 26];
        EMUSYNC;

        sd[tidDim] += sd[tidDim + 12];
        sd[tidDim + 1] += sd[tidDim + 13];
        sd[tidDim + 2] += sd[tidDim + 14];
        EMUSYNC;

        sd[tidDim] += sd[tidDim + 6];
        sd[tidDim + 1] += sd[tidDim + 7];
        sd[tidDim + 2] += sd[tidDim + 8];
        EMUSYNC;

        sd[tidDim] += sd[tidDim + 3];
        sd[tidDim + 1] += sd[tidDim + 4];
        sd[tidDim + 2] += sd[tidDim + 5];
        EMUSYNC;
    }

    if (tid == 0) {

        if (gridDim.x == 1) {

            int offset = classIndex*3;
            finalStruct[offset] = sdata[0];
            finalStruct[offset + 1] = sdata[1];
            finalStruct[offset + 2] = sdata[2];
        } else {

            int numInsIndex = classIndex * gridDim.x + blockIdx.x;
            int numInsOffset = gridDim.x * classPerRun;

            numIns[numInsIndex] = sdata[0];
            numInsIndex+=numInsOffset;
            numIns[numInsIndex] = sdata[1];
            numInsIndex+=numInsOffset;
            numIns[numInsIndex] = sdata[2];
        }

    }

}

//template < unsigned int blockSize >
__global__ void reduction6(int *entrada, int * last, int totalObjects,
                                   int arraySize, int a)
{

    unsigned int blockSize = blockDim.x;
    unsigned int classindex = blockIdx.y;
    unsigned int insIndex = blockIdx.x * blockSize * 2 + threadIdx.x;
    unsigned int realindex =
            classindex * arraySize + blockIdx.x * blockSize * 2 +
            threadIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    extern __shared__ int sdata[];

    sdata[tid] = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridSize).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread

    while (insIndex < totalObjects) {

        if (insIndex + blockSize < totalObjects) {
            sdata[tid] +=
                    entrada[realindex] + entrada[realindex +
                                                 blockSize];

        } else {
            sdata[tid] += entrada[realindex];

        }
        insIndex += gridSize;
        realindex += gridSize;
    }

    __syncthreads();

    // do reduction in shared mem
    if (blockSize == 1024 && tid < 512) {
        sdata[tid] += sdata[tid + 512];
    }
    __syncthreads();
    if (tid < 256) {
        sdata[tid] += sdata[tid + 256];
    }
    __syncthreads();
    if (tid < 128) {
        sdata[tid] += sdata[tid + 128];
    }
    __syncthreads();
    if (tid < 64) {
        sdata[tid] += sdata[tid + 64];
    }
    __syncthreads();

#ifndef __DEVICE_EMULATION__
    if (tid < 32)
#endif
    {
	volatile int *sd=sdata;
        sd[tid] += sd[tid + 32];
        EMUSYNC;
        sd[tid] += sd[tid + 16];
        EMUSYNC;
        sd[tid] += sd[tid + 8];
        EMUSYNC;
        sd[tid] += sd[tid + 4];
        EMUSYNC;
        sd[tid] += sd[tid + 2];
        EMUSYNC;
        sd[tid] += sd[tid + 1];
        EMUSYNC;

    }

    if(tid == 0) {

        if (gridDim.x == 1) {
            last[classindex*3 + a] = sdata[0];
        } else {
            entrada[classindex * arraySize + blockIdx.x] = sdata[0];
        }
    }

}
