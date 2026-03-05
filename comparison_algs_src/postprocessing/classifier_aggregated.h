#ifndef _CLASSIFIER_AGGREGATED_

#define _CLASSIFIER_AGGREGATED_

#include "classifier.h"
#include "classifier_hyperrect_list.h"
#include "JVector.h"
#include "instanceSet.h"
#include "configManagement.h"

extern configManagement cm;

class classifier_aggregated {
    JVector<classifier *> classifiers;
    int defaultClass;
    double accuracy;
    int defaultClassPolicy;

public:

    inline classifier_aggregated() {
        defaultClassPolicy = (int) cm.getParameter(DEFAULT_CLASS);
        switch (defaultClassPolicy) {
        case MAJOR:
            defaultClass = ai.getMostFrequentClass();
            //defaultClass=0;
            break;
        case MINOR:
            //defaultClass=0;
            defaultClass = ai.getLeastFrequentClass();
            break;
        case FIXED:
            defaultClass = (int) cm.getParameter(FIXED_DEFAULT_CLASS);
            break;
        case DISABLED:
        default:
            defaultClass = -1;
            break;
        }
    }

    inline ~classifier_aggregated() {
        int i;

        for (i = 0; i < classifiers.size(); i++)
            delete classifiers[i];
    }

    inline void readClassifiers(char *file) {
        defaultClass = -1;
        //JVector<classifier *>tmp;

        FILE *fp = fopen(file, "r");
        if (fp == NULL) {
            fprintf(stderr, "Cannot open rule set file %s\n", file);
            exit(1);
        }

        char string[300000];
        fgets(string, 299999, fp);
        while (!feof(fp)) {
            string[strlen(string) - 1] = 0;
            if (string[strlen(string) - 1] == 13)
                string[strlen(string) - 1] = 0;

            if (!strncmp(string, "Default rule -> ", 16)) {
                char className[100];
                strcpy(className, &string[16]);
                defaultClass = ai.valueOfNominalAttribute(
                            ai.getNumAttributesMC(), className);
                if (defaultClass == -1) {
                    fprintf(stderr, "Unknown default class %s\n", string);
                    exit(1);
                }
            } else {

                //char pheno[200000];

                classifiers.addElement(new classifier_hyperrect_list(string));
                //classifiers[classifiers.size()-1]->dumpPhenotype(pheno);
                //printf("Hey %s",pheno);
            }

            fgets(string, 299999, fp);
        }

        fclose(fp);

    }

    inline classifier_aggregated clone() {
        classifier_aggregated clone;
        JVector<classifier *> *vector = new JVector<classifier *> (
                    classifiers.size());

        clone.classifiers = *vector;
        for (int i = 0; i < classifiers.size(); i++) {
            clone.classifiers.addElement(classifiers[i]);
        }

        clone.defaultClass = defaultClass;
        clone.defaultClassPolicy = defaultClassPolicy;
        clone.accuracy = accuracy;

        return clone;

    }

    inline int getClass(int classifier) {
        if (defaultClass != -1 && classifier == classifiers.size())
            return defaultClass;
        return classifiers[classifier]->getClass();
    }

    inline int getNumClassifiers() {
        int numCL = classifiers.size();
        if (defaultClass != -1)
            numCL++;
        return numCL;
    }

    inline void setDefaultRule(instanceSet *is) {
        int i;

        if (defaultClassPolicy != DISABLED)
            return;

        int nc = ai.getNumClasses();
        int classCounts[nc];
        for (i = 0; i < nc; i++)
            classCounts[i] = 0;

        int numInst = is->getNumInstances();
        instance **instances = is->getAllInstances();
        for (i = 0; i < numInst; i++) {
            classCounts[instances[i]->instanceClass]++;
        }

        int max = classCounts[0];
        int posMax = 0;
        for (i = 1; i < nc; i++) {
            if (classCounts[i] > max) {
                posMax = i;
                max = classCounts[i];
            }
        }

        defaultClass = posMax;
    }

    inline int classify(instance *ins) {
        int i;

        int size = classifiers.size();
        for (i = 0; i < size; i++) {
            classifiers[i]->initiateEval();
            if (classifiers[i]->doMatch(ins)) {
                classifiers[i]->finalizeEval();
                return i;
            };
            classifiers[i]->finalizeEval();
        }
        if (defaultClass != -1)
            return size;
        return -1;
    }

    inline double getAccuracy() {
        return accuracy;
    }
    inline void setAccuracy(double acc) {
        accuracy = acc;
    }

    inline void dumpPhenotype(char *string) {
        int i;
        char temp[2000000];

        int size = classifiers.size();
        strcpy(string, "");
        for (i = 0; i < size; i++) {
            sprintf(temp, "%d:", i);
            strcat(string, temp);
            classifiers[i]->dumpPhenotype(temp);
            strcat(string, temp);
        }
        if (defaultClass != -1) {
            sprintf(temp, "%d:Default rule -> %s\n", i, ai.getNominalValue(
                        tGlobals->numAttributesMC, defaultClass)->cstr());
            strcat(string, temp);
        }
        strcat(string, "\n");
    }

    inline void addClassifier(classifier *cl) {
        cl->initiateEval();
        classifiers.addElement(cl);
    }

    inline int shake(int init) {

        int size = classifiers.size();
        classifier * c1 = (classifiers[init]);

        float max = 0;
        int maxindex = -1;
        classifier * maxclass;
        for (int i = init + 1; i < size; i++) {
            classifier * c2 = (classifiers[i]);
            float res = c1->calculateDistance(c2);

            if (max < res) {
                max = res;
                maxindex = i;
                maxclass = c2;
            }
        }

        printf("Max distance %f\n",max);
        if (maxindex > 0) {
            classifiers.setElementAt(maxclass, init);
            classifiers.setElementAt(c1, maxindex);
        }
        return maxindex;
    }

    inline void revertSwap(int index1, int index2) {
        classifier * c1 = classifiers[index1];
        classifier * c2 = classifiers[index2];

        classifiers.setElementAt(c1, index2);
        classifiers.setElementAt(c2, index1);
    }

    inline void removeElementAt(int index) {
        classifiers.removeElementAt(index);
    }

    inline void attributePrunning() {
        for (int i = 0; i < classifiers.size(); i++) {
            classifiers[i]=classifiers[i]->pruneAttributes();
        }
    }

    inline void attributePrunning2() {
        printf("Attribute pruning\n");
        for (int i = 0; i < classifiers.size(); i++) {
            classifiers[i]=classifiers[i]->pruneAttributes2(i);
        }
    }

    inline void attributeCleaning() {
        printf("Attribute cleaning\n");
        for (int i = 0; i < classifiers.size(); i++) {
            int change = classifiers[i]->cleanAttributes(i);
            //			if (change)
            //				printf("Changed\n");
        }
    }

    inline void attributeCleaning2() {
        printf("Attribute cleaning2\n");
        for (int i = 0; i < classifiers.size(); i++) {
            int change = classifiers[i]->cleanAttributes2(i);
            //			if (change)
            //				printf("Changed2\n");
        }
    }


};

#endif

