#include "instanceSet.h"
#include "timerMDL.h"
#include "attributesInfo.h"
#include "timerEvolutionStats.h"
#include "populationWrapper.h"
#include "utils.h"
#include "messageBuffer.h"
#include "agentPerformanceTraining.h"

extern messageBuffer mb;

extern timerEvolutionStats *tEvolStats;
extern attributesInfo ai;
extern instanceSet *is;

timerMDL::timerMDL() {

    if (!(cm.getParameter(FITNESS_FUNCTION) == MDL)) {
        mdlAccuracy = 0;
        return;
    }
    mdlAccuracy = 1;

    startIteration = (int) cm.getParameter(MDL_ITERATION);
    if (!startIteration)
        startIteration++;

    mdlWeightRelaxFactor = cm.getParameter(MDL_WEIGHT_RELAX_FACTOR);

    initialTheoryLenghtRatio = cm.getParameter(MDL_INITIAL_TL_RATIO);
    fixedWeight = 0;
    activated = 0;
    iterationMDL = 0;

    if (cm.thereIsParameter(MDL_WEIGHT)) {
        fixedWeight = 1;
        mdlWeight = cm.getParameter(MDL_WEIGHT);
    }

    coverageBreaks = new double[ai.getNumClasses()];
    adjustCovBreak(cm.getParameter(COVERAGE_BREAKPOINT));

}

void timerMDL::adjustCovBreak(double cov)
{
        int i;

        coverageBreak = cov;
        coverageRatio = cm.getParameter(COVERAGE_RATIO);
        int nc = ai.getNumClasses();
        for (i = 0; i < nc; i++) {
//                coverageBreaks[i]=coverageBreak;
                coverageBreaks[i] = coverageBreak / (double) ai.getInstancesOfClass(i)
                                * (double) is->getNumInstances();
                if (coverageBreaks[i] > 1) {
                        coverageBreaks[i] = 1;
                }
                mb.printf("Coverage break for class %d : %f\n", i, coverageBreaks[i]);
        }
}


void timerMDL::reinit() {
    fixedWeight = 0;
    activated = 0;
    iterationMDL = 0;

    if (cm.thereIsParameter(MDL_WEIGHT)) {
        fixedWeight = 1;
        mdlWeight = cm.getParameter(MDL_WEIGHT);
    }
}

void timerMDL::newIteration(int iteration, int lastIteration) {
    if (!mdlAccuracy)
        return;
    int updateWeight = 0;
    iterationMDL++;

    if (iteration == startIteration) {
        mb.printf("Iteration %d:MDL fitness activated\n", iteration);
        activated = 1;
        if (!fixedWeight) {
            classifier *ind1 = pw->getBestPopulation();
            double error = ind1->getExceptionsLength();
            double theoryLength = ind1->getTheoryLength();
            mb.printf("Error %f TL %f\n", error, theoryLength);
            if (error == 0) {
                mdlWeight = 0.1;
                fixedWeight = 1;
            } else {
                mdlWeight = (initialTheoryLenghtRatio / (1
                                                         - initialTheoryLenghtRatio)) * (error / theoryLength);
            }
        }
        updateWeight = 1;
    }

    if (activated && !fixedWeight) {
        if (pw->getBestPopulation()->getExceptionsLength() != 0) {
            if (tEvolStats->getIterationsSinceBest() == 10) {
                mdlWeight *= mdlWeightRelaxFactor;
                updateWeight = 1;
            }
        }
    }

    if (updateWeight) {
        tEvolStats->resetBestStats();
        mb.printf("MDL Theory Length Weight: %.10f (%d)\n", mdlWeight,
                  tEvolStats->getGlobalIterationsSinceBest());
        pw->activateModifiedFlag();
    }
}

void timerMDL::dumpStats(int iteration) {
    if (mdlAccuracy && activated) {
        //classifier *ind1 = pw->getBestPopulation();
        //mb.printf("Iteration %d,MDL Stats: %f %f %f\n", iteration,
        //       ind1->getTheoryLength() * mdlWeight,
        //       ind1->getExceptionsLength(),
        //       ind1->getTheoryLength() * mdlWeight /
        //       ind1->getFitness());
    }
}

double timerMDL::mdlFitness(classifier & ind, agentPerformanceTraining * ap) {
        double mdlFitness = 0;
        if (activated) {
                //printf("Valor mdlFitness %f", mdlFitness);
                mdlFitness = ind.getTheoryLength() * mdlWeight;
        }

        double exceptionsLength;
        if (ap->getNumPos() == 0) {
                exceptionsLength = 2;
        } else {
                double acc = 1 - ap->getAccuracy2();
                int cl = ind.getClass();
                ind.setRecall(ap->getRecall());
                ind.setCoverage(ap->getCoverage());
                double cov = ap->getRecall();

                ind.setRecall(cov);

                if (cov < coverageBreaks[cl] / 3) {
                        cov = 0;
                } else {
                        if (coverageBreaks[cl] < 1) {
                                if (cov < coverageBreaks[cl]) {
                                        cov = coverageRatio * cov / coverageBreaks[cl];
                                } else {
//                                        if(cov>coverageBreaks[cl]*5) cov=coverageBreaks[cl]*5;
//                                        cov=coverageRatio+(1-coverageRatio)*(cov-coverageBreaks[cl])/(1-coverageBreaks[cl]);


                                        if(cov>coverageBreaks[cl]*3) cov=coverageBreaks[cl]*3;
                                        if(cov>1) cov=1;
                                        cov = coverageRatio + (1 - coverageRatio) * (cov
                                                - coverageBreaks[cl]) / (1 - coverageBreaks[cl]);
                                }
                        }
                }

                cov = 1 - cov;// + 3*ind.getSizePercentage();
                ind.setCoverageTerm(cov);


                exceptionsLength = acc + cov;
                //printf("Acc %f Cov %f\n", acc, cov);
        }

        ind.setExceptionsLength(exceptionsLength);


        mdlFitness += exceptionsLength;
        return mdlFitness;
}


