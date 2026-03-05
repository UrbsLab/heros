#include <signal.h>
#include <time.h>
#include <sys/time.h>
#include <sys/times.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <vector>

#include "configManagement.h"
#include "populationWrapper.h"
#include "classifierFitness.h"
#include "random.h"
#include "instanceSet.h"
#include "timersManagement.h"
#include "attributesInfo.h"
#include "lex_conf.h"
#include "timeManagement.h"
#include "timerMDL.h"
#include "messageBuffer.h"
#include "classifier_aggregated.h"

using namespace std;

int stop = 0;
messageBuffer mb;
attributesInfo ai;
configManagement cm;
instanceSet *is;
timeManagement timeM;
Random rnd;
double percentageOfLearning = 0;
int lastIteration = 0;
int nodeRank;
int numTasks;

extern timerMDL *tMDL;
float minAcc = 0.5;

void handler(int sig) {
    stop = 1;
}


#ifdef __CUDA_COMPILED__
    extern "C"void setDeviceCuda();

//extern "C" void copyInstancesToDeviceReal(instance ** instances, int numInstances);
//
//extern "C" void copyInstancesToDeviceMixed(instance ** instances, int * typeOfAttributes,
//                                           int numInstances);
//
//extern "C" void copyInstancesToDeviceNominal(instance ** instances, int numInstances);
//
//extern "C" void freeAllInstanceMemory();
//
//void copyInstancesM() {
//
//    instance ** instances = is->getStratas();
//
//    if (ai.onlyRealValuedAttributes()) {
//        copyInstancesToDeviceReal(instances,
//                                  is->getNumInstances());
//        //    } else if (ai.onlyNominalAttributes() && !cm.thereIsParameter(KR_HYPERRECT)) {
//        //        copyInstancesToDeviceNominal(instances,
//        //                                     is->getNumInstances());
//    } else {
//        int * typesOfAttributes = ai.getTypeOfAttributes();
//        copyInstancesToDeviceMixed(instances, typesOfAttributes,
//                                   is->getNumInstances());
//    }
//}
//
//void freeInstanceMemoryM() {
//    freeAllInstanceMemory();
//}
#endif

classifier *runGA(timersManagement &timers) {
    stop = 0;
    lastIteration = 0;
    percentageOfLearning = 0;

    populationWrapper pw((int) cm.getParameter(POP_SIZE));

    timers.setPW(&pw);
    signal(SIGINT, handler);

    double ave1, ave2;
    pw.getAverageAccuracies(ave1, ave2);
    mb.printf("Initial population average accuracy1:%f accuracy2:%f\n", ave1,
              ave2);
    int countIt = 0, numIterations = (int) cm.getParameter(ITERATIONS);
    if (numIterations == 0)
        stop = 1;

    for (; !stop;) {
        timers.incIteration(lastIteration);

        if (is->newIteration(lastIteration)) {
            pw.activateModifiedFlag();
        }

        pw.gaIteration();

        timers.dumpStats();
        countIt++;

        if (countIt == numIterations)
            stop = 1;
        else if (countIt == numIterations - 1)
            lastIteration = 1;
        percentageOfLearning = (double) countIt / (double) (numIterations);
    }

    if (stop && countIt < numIterations)
        return NULL;

    classifier *ind = pw.getBestOverall();
    mb.printf("Best acc %f %f\n", ind->getAccuracy(), ind->getAccuracy2());
    classifier *clone = pw.cloneClassifier(ind, 0);
    //clone->postprocess();
    //clone->fitnessComputation();
    clone->adjustFitness();
    //printf("Postprocessed acc %f %f\n",clone->getAccuracy(),clone->getAccuracy2());
    return clone;
}

int main(int argc, char *argv[]) {
    int rc;

    if (argc < 4) {
        fprintf(stderr, "Incorrect parameters\n"
                "%s: <Config file> <Train file> <Test file> \n",
                argv[0]);
        exit(1);
    }



    //if (argc>=6) mb.setFile(argv[5]);

    parseConfig(argv[1]);

    rnd.dumpSeed();
    is = new instanceSet(argv[2], TRAIN);
    timersManagement timers;

    classifier_aggregated ruleSet;
    int optimizationMethod = (int) cm.getParameter(MAX_MIN);
    classifierFactory cf;
    int countRepeat = 0;


#ifdef __CUDA_COMPILED__
    printf("Codigo cuda");
    setDeviceCuda();
    //copyInstancesM();
#endif

//#ifdef __CUDA_COMPILED__
//    //freeInstanceMemoryM();
//#endif

    do {
        int cancelled = 0;
        classifier *best = NULL;

        //#ifdef __CUDA_COMPILED__
        //        //copyInstances();
        //#endif
        for (int i = 0; i < tGlobals->numRepetitionsLearning; i++) {
            classifier *bestIt = runGA(timers);
            if (!bestIt) {
                cancelled = 1;
                break;
            }

            if (best == NULL || bestIt->compareToIndividual(best,
                                                            optimizationMethod) > 0) {
                if (best)
                    cf.deleteClassifier(best);
                best = bestIt;
            }

            if (i < tGlobals->numRepetitionsLearning - 1) {
                is->restart();
                timers.reinit();
            }

        }
        //#ifdef __CUDA_COMPILED__
        //        freeInstanceMemory();
        //#endif
        if (cancelled)
            break;


        if (isMajority(*best)) {
            char phenotype[2000000];
            best->dumpPhenotype(phenotype);
            mb.printf("Rule:%s", phenotype);
            is->removeInstancesAndRestart(best);
            ruleSet.addClassifier(best);
            classifierBriefTest(ruleSet, is);
            countRepeat = 0;
            timers.reinit();
            if (is->getNumInstances() == 0)
                break;
        } else {
            countRepeat++;
            cf.deleteClassifier(best);
            if (countRepeat == 3) {
                ruleSet.setDefaultRule(is);
                break;
            } else {
                is->restart();
                timers.reinit();
            }
        }
        //#ifdef __CUDA_COMPILED__
        //        copyInstances();
        //#endif

    } while (1);

    delete is;

    char phenotype[2000000];
    ruleSet.dumpPhenotype(phenotype);
    mb.printf("Phenotype: \n%s\n", phenotype);


    classifierStats(ruleSet, argv[2], "Train");
    classifierStats(ruleSet, argv[3], "Test");

    return 0;
}
