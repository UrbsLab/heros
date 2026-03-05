#include "classifier_aggregated.h"
#include "postprocessingOper.h"

extern configManagement cm;

void applyPostProcessing(classifier_aggregated & ind) {

    int oper[6] = {OPERATOR1_POLICY,OPERATOR2_POLICY,OPERATOR3_POLICY,OPERATOR4_POLICY,OPERATOR5_POLICY,OPERATOR6_POLICY};

    for(int i=0; i<6; i++) {
        if(cm.thereIsParameter(oper[i])) {
            switch ((int) cm.getParameter(oper[i])) {
            case CL:
                ind.attributeCleaning();
                break;
            case CL2:
                ind.attributeCleaning2();
                break;
            case PR:
                ind.attributePrunning2();
                break;
            case SW:
                ruleSetShake(ind);
                break;
            case NONE:
            default:
                break;

            }
        }
    }
}

void attributePrunning(classifier_aggregated & ind) {
    ind.attributePrunning();
}

void attributePrunning2(classifier_aggregated & ind) {
    ind.attributePrunning2();
}

void attributeCleaning(classifier_aggregated & ind) {
    ind.attributeCleaning();
}

void attributeCleaning2(classifier_aggregated & ind) {
    ind.attributeCleaning2();
}

void ruleSetShake(classifier_aggregated & ind) {

    agentPerformance ap = calculateAgentPerformance(ind);

    ind.setAccuracy(ap.getAccuracy());
    int initial = ind.getNumClassifiers();
    char phenotype[2000000];

    double accuracy = ap.getAccuracy();

    printf("Rule swapping\n");

    int n = ind.getNumClassifiers() - 1;
    for (int i = 0; i < n - 1; i++) {
        int index = ind.shake(i);

        printf("Trying swap %d and %d\n", i, index);

        if (index > 0) {
            agentPerformance ap2 = calculateAgentPerformance(ind);

            if (ap2.getAccuracy() < accuracy) {
                ind.revertSwap(i, index);
            } else {
                printf("swap between %d and %d\n", i, index);
                int m = n;
                JVector<int> removedIndex;

                for (int j = i; j < m; j++) {
                    if (ap2.getActivationsOfClassifier(j) <= 0) {
                        removedIndex.addElement(j);
                    }
                }

                for (int j = 0; j < removedIndex.size(); j++) {
                    printf("Eliminando classificador %d\n", removedIndex[j]);
                    ind.removeElementAt(removedIndex[j]-j);
                    n--;
                }

                accuracy = ap2.getAccuracy();

            }

        }
    }

    ind.dumpPhenotype(phenotype);
    mb.printf("Phenotype after rule swap: \n%s\n", phenotype);
    printf("Number of rules: Initial %d Final %d\n", initial,
           ind.getNumClassifiers());
    //delete is;
}
