#include <signal.h>
#include <time.h>
#include <sys/time.h>
#include <sys/times.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>

#include "configManagement.h"
#include "populationWrapper.h"
#include "classifierFitness.h"
#include "postprocessingOper.h"
#include "random.h"
#include "instanceSet.h"
#include "timersManagement.h"
#include "attributesInfo.h"
#include "lex_conf.h"
#include "timeManagement.h"
#include "timerMDL.h"
#include "messageBuffer.h"
#include "classifier_aggregated.h"
#include "classifier.h"

using namespace std;

int stop = 0;
messageBuffer mb;
attributesInfo ai;
configManagement cm;
instanceSet *is;
instanceSet *isTrain;
instanceSet *isTest;
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



int main(int argc, char *argv[]) {

    if (argc < 4) {
        fprintf(stderr, "Incorrect parameters\n"
                "%s: <Config file> <rules> <Train file> [Test file] \n", argv[0]);
        exit(1);
    }

    parseConfig(argv[1]);

    rnd.dumpSeed();
    is = new instanceSet(argv[3], TRAIN);

    timersManagement timers;

    classifier_aggregated ruleSet;
    ruleSet.readClassifiers(argv[2]);

    char phenotype[2000000];

    // Printing statistics of train and test before altering
    // the classifiers
    if(cm.thereIsParameter(TRAIN_STATS_ENABLED)) {
        int val = (int) cm.getParameter(TRAIN_STATS_ENABLED);
        isTrain = new instanceSet(argv[3], TEST);
        if(val == ALL || val == START) {
            classifierStats(ruleSet, isTrain, "Train");
        }
    }

    if(argc >= 5 && cm.thereIsParameter(TEST_STATS_ENABLED)) {
        int val = (int) cm.getParameter(TEST_STATS_ENABLED);
        isTest = new instanceSet(argv[4], TEST);
        if(val == ALL || val == START) {
            classifierStats(ruleSet, isTest, "Test");
        }
    }
    //Apply operators
    applyPostProcessing(ruleSet);

    // Printing statistics of train and test after
    // applying the operators
    if(cm.thereIsParameter(TRAIN_STATS_ENABLED)) {
        int val = (int) cm.getParameter(TRAIN_STATS_ENABLED);
        if(val == ALL || val == END) {
            classifierStats(ruleSet, isTrain, "Train");
        }
    }

    if(argc >= 5 && cm.thereIsParameter(TEST_STATS_ENABLED)) {
        int val = (int) cm.getParameter(TEST_STATS_ENABLED);
        if(val == ALL || val == END) {
            classifierStats(ruleSet, isTest, "Test");
        }
    }

    //Printing new set of rules
    ruleSet.dumpPhenotype(phenotype);
    mb.printf("Phenotype: \n%s\n", phenotype);

    delete is;
    delete isTest;
    delete isTrain;

    return 0;
}
