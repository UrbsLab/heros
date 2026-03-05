#include "classifier_hyperrect_list.h"
#include "random.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <strings.h>
#include "attributesInfo.h"
#include "timerGlobals.h"
#include "timerMutation.h"
#include "timerCrossover.h"
#include "timerMDL.h"
#include "timerHierar.h"
#include "timerRealKR.h"
#include "instanceSet.h"

#include <algorithm>

extern attributesInfo ai;
extern timerGlobals *tGlobals;
extern timerHierar *tHierar;
extern timerMutation *tMut;
extern timerCrossover *tCross;
extern timerMDL *tMDL;
extern timerRealKR *tReal;
extern instanceSet *is;
extern Random rnd;

double classifier_hyperrect_list::computeTheoryLength() {
    int i, j, base;
    theoryLength = 0.0;

    float *ptr = predicates;

    for (i = 0; i < numAtt; i++) {
        int att = whichAtt[i];
        if (ai.getTypeOfAttribute(att) == REAL) {
            float size = ai.getSizeDomain(att);
            if (size > 0) {
                theoryLength += 1.0 - (ptr[1] - ptr[0]) / size;
            }
        } else {
            double countFalses = 0;
            int numValues = tReal->attributeSize[att];
            for (j = 0; j < numValues; j++) {
                if (!ptr[j])
                    countFalses++;
            }
            theoryLength += (double) countFalses / (double) numValues;
        }
        ptr += tReal->attributeSize[att];
    }
    theoryLength /= (double) tGlobals->numAttributesMC;

    return theoryLength;
}

classifier_hyperrect_list::classifier_hyperrect_list(char *string) {
    //JVector<term *> termsList;

    JVector<float> pred;
    JVector<int> attributeList;
    JVector<int> offsets;
    int offset = 0;
    char *token;
    int numAttributes = ai.getNumAttributesMC();
    char *saveptr1;

    token = strtok_r(string, "|", &saveptr1);
    int inloop = 1;
    while (inloop && token != NULL) {
        if (!strncmp(token, "Att ", 4)) {
            char *saveptr2;
            char *token2 = strtok_r(token, " ", &saveptr2);
            if (token2 == NULL || strcmp(token2, "Att")) {
                fprintf(stderr, "Wrong term1 %s\n", token);
                exit(1);
            }
            token2 = strtok_r(NULL, " ", &saveptr2);
            if (token2 == NULL) {
                fprintf(stderr, "Wrong term2 %s\n", token);
                exit(1);
            }
            int attIndex = ai.getAttributeIndex(token2);
            offsets.addElement(offset);
            if (attIndex == -1) {
                fprintf(stderr, "Unknown attribute name %s\n", token2);
                exit(1);
            }
            attributeList.addElement(attIndex);
            token2 = strtok_r(NULL, " ", &saveptr2);
            if (token2 == NULL || strcmp(token2, "is")) {
                fprintf(stderr, "Wrong term3 %s\n", token);
                exit(1);
            }
            token2 = strtok_r(NULL, " ", &saveptr2);
            if (token2 == NULL) {
                fprintf(stderr, "Wrong term4 %s\n", token);
                exit(1);
            }
            if (ai.getTypeOfAttribute(attIndex) == REAL) {
                //term *realTerm;
                offset += 2;
                float lb, ub;
                if (sscanf(token2, "[%f,%f]", &lb, &ub) == 2) {
                    pred.addElement(lb);
                    pred.addElement(ub);
                    //realTerm = new term1(lb,ub);
                } else if (sscanf(token2, "[>%f]", &lb) == 1) {
                    pred.addElement(lb);
                    pred.addElement(ai.getMaxDomain(attIndex));
                    //realTerm = new term3(lb);
                } else if (sscanf(token2, "[<%f]", &ub) == 1) {
                    pred.addElement(ai.getMinDomain(attIndex));
                    pred.addElement(ub);
                    //realTerm = new term2(ub);
                } else {
                    fprintf(stderr,
                            "Wrong real prediacte %s for attribute %d\n",
                            token2, attIndex);
                    exit(1);
                }
                //termsList.addElement(realTerm);
            } else {
                int numValues = ai.getNumValuesAttribute(attIndex);
                offset += numValues;
                int *predicate = new int[numValues];
                bzero(predicate, numValues * sizeof(int));
                char *saveptr3;
                char *token3 = strtok_r(token2, ",", &saveptr3);
                while (token3 != NULL) {
                    int value = ai.valueOfNominalAttribute(attIndex, token3);
                    if (value == -1) {
                        fprintf(stderr, "Wrong value %s for attribute %d\n",
                                token3, attIndex);
                        exit(1);
                    }
                    predicate[value] = 1;
                    token3 = strtok_r(NULL, ",", &saveptr3);
                }

                for (int h = 0; h < numValues; h++) {
                    pred.addElement((float) predicate[h]);
                }
                //termsList.addElement(new termNominal(predicate,numValues));
            }
        } else {
            classValue = ai.valueOfNominalAttribute(ai.getNumAttributesMC(),
                                                    token);
            if (classValue == -1) {
                fprintf(stderr, "Wrong class %s\n", token);
                exit(1);
            }
            inloop = 0;
        }
        token = strtok_r(NULL, "|", &saveptr1);
    }

    numAtt = attributeList.size();
    whichAtt = new int[numAtt];
    //terms = new term *[numAtt];

    for (int i = 0; i < numAtt; i++) {
        int att = attributeList[i];
        whichAtt[i] = att;
    }

    ruleSize = pred.size();
    predicates = new float[ruleSize];
    for (int i = 0; i < ruleSize; i++) {
        predicates[i] = pred[i];
    }

    offsetPredicates = new int[offsets.size()];
    for (int i = 0; i < offsets.size(); i++) {
        offsetPredicates[i] = offsets[i];
    }
}

classifier_hyperrect_list::classifier_hyperrect_list(int empty) {
    classifier_hyperrect_list::initializeChromosome(empty);
}

classifier_hyperrect_list::classifier_hyperrect_list(
    const classifier_hyperrect_list & orig, int son) {
    *this = orig;

    if (!son) {
        whichAtt = new int[numAtt];
        bcopy(orig.whichAtt, whichAtt, numAtt * sizeof(int));

        offsetPredicates = new int[numAtt];
        bcopy(orig.offsetPredicates, offsetPredicates, numAtt * sizeof(int));

        predicates = new float[ruleSize];
        bcopy(orig.predicates, predicates, ruleSize * sizeof(float));
    } else {
        whichAtt = NULL;
        predicates = NULL;
        offsetPredicates = NULL;
    }
}

classifier_hyperrect_list::~classifier_hyperrect_list() {
    delete whichAtt;
    delete predicates;
    delete offsetPredicates;
}

void classifier_hyperrect_list::initializeChromosome(int empty) {
    int i, j, base;

    instance *ins = NULL;
    if (tGlobals->smartInit) {
        if (tGlobals->defaultClassPolicy != DISABLED) {
            ins = is->getInstanceInit(tGlobals->defaultClass);
        } else {
            ins = is->getInstanceInit(ai.getNumClasses());
        }
    }

    JVector<int> selectedAtts;
    ruleSize = 0;
    if (!empty) {
        for (i = 0; i < tGlobals->numAttributesMC; i++) {
            if (!rnd >= tReal->probIrr) {
                selectedAtts.addElement(i);
                ruleSize += tReal->attributeSize[i];
            }
        }
    }

    if (!empty) {
        if (ins) {
            classValue = ins->getClass();
        } else {
            do {
                classValue = rnd(0, ai.getNumClasses() - 1);
            } while (tGlobals->defaultClassPolicy != DISABLED
                     && classValue == tGlobals->defaultClass);
        }
    } else {
        if (tGlobals->defaultClassPolicy != DISABLED) {
            classValue = is->getMajorityClassExcept(tGlobals->defaultClass);
        } else {
            classValue = is->getMajorityClass();
        }
    }

    numAtt = selectedAtts.size();
    whichAtt = new int[numAtt];
    offsetPredicates = new int[numAtt];
    predicates = new float[ruleSize];

    for (i = 0, base = 0; i < numAtt; i++) {
        offsetPredicates[i] = base;
        int att = selectedAtts[i];
        whichAtt[i] = att;

        if (ai.getTypeOfAttribute(att) == REAL) {
            float max, min;
            float sizeD = ai.getSizeDomain(att);
            float minD = ai.getMinDomain(att);
            float maxD = ai.getMaxDomain(att);
            float size = (!rnd * 0.5 + 0.25) * sizeD;

            if (ins) {
                float val = ins->realValues[att];
                min = val - size / 2.0;
                max = val + size / 2.0;
                if (min < minD) {
                    max += (minD - min);
                    min = minD;
                }
                if (max > maxD) {
                    min -= (max - maxD);
                    max = maxD;
                }
            } else {
                min = !rnd * (sizeD - size) + minD;
                max = min + size;
            }

            predicates[base] = min;
            predicates[base + 1] = max;
        } else {
            int value;
            if (ins)
                value = (unsigned char) ins->realValues[att];
            else
                value = -1;
            for (j = 0; j < tReal->attributeSize[att]; j++) {
                if (j != value) {

                    //					int insSum=0;
                    //					double sum = 0;
                    //					for(int k=0; k < ai.getNumClasses();k++) {
                    //						insSum += ai.getInstancesOfClass(k);
                    //						sum += ai.getFrequencyOfValueOfAttribute(k,att,j)*ai.getInstancesOfClass(k);
                    //					}

                    double sum = (1 / (double) tReal->attributeSize[att])
                            / ai.getFrequencyOfValueOfAttribute(classValue, att,
                                                                j);
                    //sum /= insSum;
                    //printf("Especificity constant att %d val %d class %d: %f %f\n",att,j,classValue,sum,min(1.0, sum*tGlobals->probOne));

                    //if(!rnd< min(1.0, sum*tGlobals->probOne)) {

                    if (!rnd < tGlobals->probOne) {
                        predicates[base + j] = 1;
                    } else {
                        predicates[base + j] = 0;
                    }
                    //printf("Adding %d %d %d : %d\n", att,j,classValue, (int) predicates[base+j]);
                } else {
                    predicates[base + j] = 1;
                    //printf("Adding+ %d %d %d : %d\n", att,j,classValue, (int) predicates[base+j]);
                }
            }
        }

        base += tReal->attributeSize[att];
    }

}

void classifier_hyperrect_list::crossover(classifier * in, classifier * out1,
                                          classifier * out2) {
    crossover_1px(this, (classifier_hyperrect_list *) in,
                  (classifier_hyperrect_list *) out1,
                  (classifier_hyperrect_list *) out2);
}

float classifier_hyperrect_list::mutationOffset(float geneValue,
                                                float offsetMin, float offsetMax) {
    float newValue;
    if (!rnd < 0.5) {
        newValue = geneValue + !rnd * offsetMax;
    } else {
        newValue = geneValue - !rnd * offsetMin;
    }
    return newValue;
}

void classifier_hyperrect_list::mutation() {
    int i;
    int attribute, value, attIndex;

    modif = 1;

    if (tGlobals->numClasses > 1 && !rnd < 0.10) {
        int newValue;
        do {
            newValue = rnd(0, ai.getNumClasses() - 1);
        } while (newValue == classValue
                 || tGlobals->defaultClassPolicy != DISABLED
                 && newValue == tGlobals->defaultClass);
        classValue = newValue;
    } else {
        if (numAtt > 0) {
            attIndex = rnd(0, numAtt - 1);
            attribute = whichAtt[attIndex];
            value = rnd(0, tReal->attributeSize[attribute] - 1);
            int index = offsetPredicates[attIndex] + value;

            if (ai.getTypeOfAttribute(attribute) == REAL) {
                float newValue, minOffset, maxOffset;
                minOffset = maxOffset = 0.5 * ai.getSizeDomain(attribute);
                newValue = mutationOffset(predicates[index], minOffset,
                                          maxOffset);
                if (newValue < ai.getMinDomain(attribute))
                    newValue = ai.getMinDomain(attribute);
                if (newValue > ai.getMaxDomain(attribute))
                    newValue = ai.getMaxDomain(attribute);
                predicates[index] = newValue;
                if (value)
                    index--;
                if (predicates[index] > predicates[index + 1]) {
                    swapD(predicates[index], predicates[index + 1]);
                }
            } else {
                if (predicates[index] == 1)
                    predicates[index] = 0;
                else
                    predicates[index] = 1;
            }
        }
    }
}

void classifier_hyperrect_list::dumpPhenotype(char *string) {
    char temp[10000];
    char temp2[10000];
    char tmp1[20];
    char tmp2[20];
    int i, j, index;

    strcpy(string, "");

    for (i = 0, index = 0; i < numAtt; i++) {
        int attIndex = whichAtt[i];
        sprintf(temp, "Att %s is ", ai.getAttributeName(attIndex)->cstr());
        int irr = 1;

        if (ai.getTypeOfAttribute(attIndex) == REAL) {
            float minD = ai.getMinDomain(attIndex);
            float maxD = ai.getMaxDomain(attIndex);
            if (predicates[index] == minD) {
                if (predicates[index + 1] == maxD) {
                    // do nothing
                } else {
                    irr = 0;
                    sprintf(temp2, "[<%f]", predicates[index + 1]);
                    strcat(temp, temp2);
                }
            } else {
                if (predicates[index + 1] == maxD) {
                    irr = 0;
                    sprintf(temp2, "[>%f]", predicates[index]);
                    strcat(temp, temp2);
                } else {
                    irr = 0;
                    sprintf(tmp1, "%f", predicates[index]);
                    sprintf(tmp2, "%f", predicates[index + 1]);
                    sprintf(temp2, "[%s,%s]", tmp1, tmp2);
                    strcat(temp, temp2);
                }
            }
        } else {
            for (j = 0; j < tReal->attributeSize[attIndex]; j++) {
                if (predicates[index + j]) {
                    sprintf(temp2, "%s,",
                            ai.getNominalValue(attIndex, j)->cstr());
                    strcat(temp, temp2);
                } else {
                    irr = 0;
                }
            }
            if (temp[strlen(temp) - 1] == ',')
                temp[strlen(temp) - 1] = 0;
        }

        index += tReal->attributeSize[attIndex];

        if (!irr) {
            strcat(string, temp);
            strcat(string, "|");
        }
    }
    sprintf(temp, "%s\n",
            ai.getNominalValue(tGlobals->numAttributesMC, classValue)->cstr());
    strcat(string, temp);
}

void classifier_hyperrect_list::crossover_1px(classifier_hyperrect_list * in1,
                                              classifier_hyperrect_list * in2, classifier_hyperrect_list * out1,
                                              classifier_hyperrect_list * out2) {
    int i;

    out1->modif = out2->modif = 1;

    if (in1->numAtt == 0) {
        classifier_hyperrect_list *tmp = in2;
        in2 = in1;
        in1 = tmp;
    }

    if (in1->numAtt == 0) {
        out1->whichAtt = new int[out1->numAtt];
        out2->whichAtt = new int[out2->numAtt];
        out1->offsetPredicates = new int[out1->numAtt];
        out2->offsetPredicates = new int[out2->numAtt];
        out1->predicates = new float[out1->ruleSize];
        out2->predicates = new float[out2->ruleSize];
        return;
    }

    int pos1 = rnd(0, in1->numAtt - 1);
    int selAtt1 = in1->whichAtt[pos1];

    for (i = 0; i < in2->numAtt && in2->whichAtt[i] < selAtt1; i++)
        ;
    int pos2 = i;
    int selAtt2;
    if (pos2 != in2->numAtt) {
        selAtt2 = in2->whichAtt[pos2];
    } else {
        selAtt2 = -1;
    }

    out1->numAtt = pos1 + 1 + (in2->numAtt - pos2);
    out2->numAtt = pos2 + (in1->numAtt - pos1 - 1);
    if (selAtt1 == selAtt2) {
        out1->numAtt--;
        out2->numAtt++;
    }

    out1->whichAtt = new int[out1->numAtt];
    out2->whichAtt = new int[out2->numAtt];
    out1->offsetPredicates = new int[out1->numAtt];
    out2->offsetPredicates = new int[out2->numAtt];

    out1->ruleSize = 0;
    for (i = 0; i <= pos1; i++) {
        out1->whichAtt[i] = in1->whichAtt[i];
        out1->offsetPredicates[i] = out1->ruleSize;
        out1->ruleSize += tReal->attributeSize[out1->whichAtt[i]];
    }
    int lenp1c1 = out1->ruleSize;
    int base = pos2;
    if (selAtt1 == selAtt2) {
        base++;
    }
    for (; i < out1->numAtt; i++, base++) {
        out1->whichAtt[i] = in2->whichAtt[base];
        out1->offsetPredicates[i] = out1->ruleSize;
        out1->ruleSize += tReal->attributeSize[out1->whichAtt[i]];
    }

    out2->ruleSize = 0;
    for (i = 0; i < pos2; i++) {
        out2->whichAtt[i] = in2->whichAtt[i];
        out2->offsetPredicates[i] = out2->ruleSize;
        out2->ruleSize += tReal->attributeSize[out2->whichAtt[i]];
    }
    int lenp2c1 = out2->ruleSize;
    base = pos1;
    if (selAtt1 != selAtt2) {
        base++;
    }
    for (; i < out2->numAtt; i++, base++) {
        out2->whichAtt[i] = in1->whichAtt[base];
        out2->offsetPredicates[i] = out2->ruleSize;
        out2->ruleSize += tReal->attributeSize[out2->whichAtt[i]];
    }

    out1->predicates = new float[out1->ruleSize];
    out2->predicates = new float[out2->ruleSize];

    bcopy(in1->predicates, out1->predicates, lenp1c1 * sizeof(float));
    bcopy(in2->predicates, out2->predicates, lenp2c1 * sizeof(float));

    if (selAtt1 == selAtt2) {
        int baseP1 = in1->offsetPredicates[pos1];
        int baseP2 = in2->offsetPredicates[pos2];
        int baseO1 = out1->offsetPredicates[pos1];
        int baseO2 = out2->offsetPredicates[pos2];

        if (ai.getTypeOfAttribute(selAtt1) == REAL) {
            int cutPoint = rnd(0, 2);
            if (cutPoint == 0) {
                out1->predicates[baseO1] = in2->predicates[baseP2];
                out1->predicates[baseO1 + 1] = in2->predicates[baseP2 + 1];
                out2->predicates[baseO2] = in1->predicates[baseP1];
                out2->predicates[baseO2 + 1] = in1->predicates[baseP1 + 1];
            } else if (cutPoint == 1) {
                float min1 = in1->predicates[baseP1];
                float min2 = in2->predicates[baseP2];
                float max1 = in2->predicates[baseP2 + 1];
                float max2 = in1->predicates[baseP1 + 1];

                if (min1 > max1)
                    swapD(min1, max1);
                if (min2 > max2)
                    swapD(min2, max2);

                out1->predicates[baseO1] = min1;
                out1->predicates[baseO1 + 1] = max1;
                out2->predicates[baseO2] = min2;
                out2->predicates[baseO2 + 1] = max2;
            } else {
                out1->predicates[baseO1] = in1->predicates[baseP1];
                out1->predicates[baseO1 + 1] = in1->predicates[baseP1 + 1];
                out2->predicates[baseO2] = in2->predicates[baseP2];
                out2->predicates[baseO2 + 1] = in2->predicates[baseP2 + 1];
            }
        } else {
            int size = tReal->attributeSize[selAtt1];
            int cutPoint = rnd(0, size);
            bcopy(&in1->predicates[baseP1], &out1->predicates[baseO1],
                  cutPoint * sizeof(float));
            bcopy(&in2->predicates[baseP2], &out2->predicates[baseO2],
                  cutPoint * sizeof(float));

            bcopy(&in1->predicates[baseP1 + cutPoint],
                  &out2->predicates[baseO2 + cutPoint],
                  (size - cutPoint) * sizeof(float));
            bcopy(&in2->predicates[baseP2 + cutPoint],
                  &out1->predicates[baseO1 + cutPoint],
                  (size - cutPoint) * sizeof(float));
        }
        pos2++;
    } else {
        int base1 = in1->offsetPredicates[pos1];
        out1->whichAtt[pos1] = selAtt1;
        bcopy(&in1->predicates[base1], &out1->predicates[base1],
              tReal->attributeSize[selAtt1] * sizeof(float));
    }

    pos1++;
    if (pos1 < in1->numAtt) {
        bcopy(&in1->predicates[in1->offsetPredicates[pos1]],
              &out2->predicates[out2->offsetPredicates[pos2]],
              (in1->ruleSize - in1->offsetPredicates[pos1]) * sizeof(float));
    }
    if (pos2 < in2->numAtt) {
        bcopy(&in2->predicates[in2->offsetPredicates[pos2]],
              &out1->predicates[out1->offsetPredicates[pos1]],
              (in2->ruleSize - in2->offsetPredicates[pos2]) * sizeof(float));
    }

    if (!rnd < 0.5) {
        out1->classValue = in1->classValue;
        out2->classValue = in2->classValue;
    } else {
        out1->classValue = in2->classValue;
        out2->classValue = in1->classValue;
    }
}

void classifier_hyperrect_list::postprocess() {
}

void classifier_hyperrect_list::deleteAttribute(int attribute) {

    int i;
    int deletedSize = tReal->attributeSize[whichAtt[attribute]];
    int *newWhichAtt = new int[numAtt - 1];
    int *newOffsetPredicates = new int[numAtt - 1];
    float *newPredicates = new float[ruleSize - deletedSize];
    bcopy(whichAtt, newWhichAtt, attribute * sizeof(int));
    bcopy(offsetPredicates, newOffsetPredicates, attribute * sizeof(int));
    bcopy(predicates, newPredicates,
          offsetPredicates[attribute] * sizeof(float));
    if (attribute != numAtt - 1) {
        bcopy(&whichAtt[attribute + 1], &newWhichAtt[attribute],
              (numAtt - attribute - 1) * sizeof(int));
        bcopy(&offsetPredicates[attribute + 1], &newOffsetPredicates[attribute],
              (numAtt - attribute - 1) * sizeof(int));
        bcopy(&predicates[offsetPredicates[attribute + 1]],
              &newPredicates[offsetPredicates[attribute]],
              (ruleSize - offsetPredicates[attribute + 1]) * sizeof(float));
    }
    delete whichAtt;
    whichAtt = newWhichAtt;
    delete offsetPredicates;
    offsetPredicates = newOffsetPredicates;
    delete predicates;
    predicates = newPredicates;
    numAtt--;
    ruleSize -= deletedSize;
    for (i = attribute; i < numAtt; i++) {
        offsetPredicates[i] -= deletedSize;
    }
}

void classifier_hyperrect_list::specifyAttribute(int selectedAtt) {

    int i;
    int addedSize = tReal->attributeSize[selectedAtt];
    int *newWhichAtt = new int[numAtt + 1];
    int *newOffsetPredicates = new int[numAtt + 1];
    float *newPredicates = new float[ruleSize + addedSize];
    int index = 0, index2 = 0;
    while (index < numAtt && whichAtt[index] < selectedAtt) {
        newWhichAtt[index] = whichAtt[index];
        newOffsetPredicates[index] = offsetPredicates[index];
        bcopy(&predicates[index2], &newPredicates[index2],
              tReal->attributeSize[whichAtt[index]] * sizeof(float));
        index2 += tReal->attributeSize[whichAtt[index]];
        index++;
    }
    newWhichAtt[index] = selectedAtt;
    newOffsetPredicates[index] = index2;
    if (ai.getTypeOfAttribute(selectedAtt) == REAL) {
        float sizeD = ai.getSizeDomain(selectedAtt);
        float minD = ai.getMinDomain(selectedAtt);
        float maxD = ai.getMaxDomain(selectedAtt);
        float size = (!rnd * 0.5 + 0.25) * sizeD;
        float min = !rnd * (sizeD - size) + minD;
        float max = min + size;
        newPredicates[index2] = min;
        newPredicates[index2 + 1] = max;
    } else {
        for (i = 0; i < tReal->attributeSize[selectedAtt]; i++) {
            if (!rnd < tGlobals->probOne) {
                newPredicates[index2 + i] = 1;
            } else {
                newPredicates[index2 + i] = 0;
            }
        }

    }
    if (index != numAtt) {
        bcopy(&whichAtt[index], &newWhichAtt[index + 1],
              (numAtt - index) * sizeof(int));
        bcopy(&offsetPredicates[index], &newOffsetPredicates[index + 1],
              (numAtt - index) * sizeof(int));
        bcopy(&predicates[index2], &newPredicates[index2 + addedSize],
              (ruleSize - offsetPredicates[index]) * sizeof(float));
    }
    delete whichAtt;
    whichAtt = newWhichAtt;
    delete offsetPredicates;
    offsetPredicates = newOffsetPredicates;
    delete predicates;
    predicates = newPredicates;
    numAtt++;
    ruleSize += addedSize;
    for (i = index + 1; i < numAtt; i++) {
        offsetPredicates[i] += addedSize;
    }
}

void classifier_hyperrect_list::doSpecialStage(int stage) {
    int i;

    if (stage == 0) { //Generalize
        if (numAtt > 0 && !rnd < tReal->probGeneralizeList) {
            int attribute = rnd(0, numAtt - 1);
            deleteAttribute(attribute);
        }
    } else { //Specialize
        if (numAtt < tGlobals->numAttributesMC
                && !rnd < tReal->probSpecializeList) {
            int attMap[tGlobals->numAttributesMC];
            bzero(attMap, tGlobals->numAttributesMC * sizeof(int));
            for (i = 0; i < numAtt; i++) {
                attMap[whichAtt[i]] = 1;
            }

            int selectedAtt;
            do {
                selectedAtt = rnd(0, tGlobals->numAttributesMC - 1);
            } while (attMap[selectedAtt] == 1);
            specifyAttribute(selectedAtt);
        }
    }
}

void classifier_hyperrect_list::initiateEval() {
    int i;
    JVector<int> discrete;
    JVector<int> real;

    for (i = 0; i < numAtt; i++) {
        int att = whichAtt[i];
        if (ai.getTypeOfAttribute(att) == REAL) {
            real.addElement(i);
        } else {
            discrete.addElement(i);
        }
    }

    numReal = real.size();
    listRealPos = new int[numReal];
    listRealAtt = new int[numReal];
    for (i = 0; i < numReal; i++) {
        listRealPos[i] = real[i];
        listRealAtt[i] = whichAtt[listRealPos[i]];
    }

    numDiscrete = discrete.size();
    listDiscretePos = new int[numDiscrete];
    listDiscreteAtt = new int[numDiscrete];
    for (i = 0; i < numDiscrete; i++) {
        listDiscretePos[i] = discrete[i];
        listDiscreteAtt[i] = whichAtt[listDiscretePos[i]];
    }
}

double classifier_hyperrect_list::getAttributeSpecificity(int i) {
    int base = offsetPredicates[i];
    int att = whichAtt[i];
    int totVals = ai.getNumValuesAttribute(att);
    float result = 0;
    for (int j = 0; j < totVals; j++) {
        result += predicates[base + j];

    }
    float result2 = result / ((double) totVals);
    return result2;

}

int classifier_hyperrect_list::getNumAttributesSpecified() {
    int res = 0;
    for (int i = 0; i < numAtt; i++) {
        if (getAttributeSpecificity(i) < 1)
            res += 1;
    }
    return res;
}

double classifier_hyperrect_list::getAttributesSpecificity() {

    float res = 0;
    int a = 0;
    for (int i = 0; i < numAtt; i++) {
        if (getAttributeSpecificity(i) != 1) {
            res += getAttributeSpecificity(i);
            a++;
        }
    }
    return res / (double) a;
}

classifier_hyperrect_list *classifier_hyperrect_list::pruneAttributes() {
    classifier_hyperrect_list * orig = new classifier_hyperrect_list(*this, 0);
    orig->fitnessComputation();
    double acc = orig->getAccuracy2();
    char pheno[1000000];

    orig->dumpPhenotype(pheno);
    printf("Class Acc %f Cov %f Matched %d Pos %d Ok %d %s", acc,
           orig->getCoverage(), orig->getNumInstancesMatched(),
           orig->getNumInstancesPos(), orig->getNumInstancesPosOK(), pheno);
    instance ** insIter = is->getInstancesOfIteration();

    for (int i = 0; i < orig->numAtt; i++) {
        if (!(orig->getAttributeSpecificity(i) == 1.0)) {
            classifier_hyperrect_list * copy3 = new classifier_hyperrect_list(
                        *orig, 0);
            copy3->deleteAttribute(i);
            copy3->fitnessComputation();
            copy3->dumpPhenotype(pheno);

            printf("acc %f %f Cov %f Matched %d Pos %d Ok %d %s", acc,
                   copy3->getAccuracy2(), copy3->getCoverage(),
                   copy3->getNumInstancesMatched(),
                   copy3->getNumInstancesPos(), copy3->getNumInstancesPosOK(),
                   pheno);
            if (!(copy3->getAccuracy2() < acc)) {
                printf("Eliminado attributo %d\n", i);
                i--;
                orig = copy3;

                if (copy3->getAccuracy2() > acc)
                    acc = copy3->getAccuracy2();
            }

        }

    }

    return orig;
}

//Do not accept covering any new negative examples!
classifier_hyperrect_list *classifier_hyperrect_list::pruneAttributes2(int index) {
    classifier_hyperrect_list * orig = new classifier_hyperrect_list(*this, 0);
    orig->fitnessComputation();
    double acc = orig->getAccuracy2();
    double cov = orig->getCoverage();
    //char pheno[1000000];

    //orig->dumpPhenotype(pheno);
    int origmatched = orig->getNumInstancesMatched();
    int origok = orig->getNumInstancesPosOK();
    //printf("Class Acc %f Cov %f Matched %d Pos %d Ok %d %s", acc,
    //		orig->getCoverage(), origmatched, orig->getNumInstancesPos(),
    //		origok, pheno);

    int atts = 0;
    for (int i = 0; i < orig->numAtt; i++) {

        if (!(orig->getAttributeSpecificity(i) == 1.0)) {
            classifier_hyperrect_list * copy3 = new classifier_hyperrect_list(
                        *orig, 0);
            copy3->deleteAttribute(i);
            copy3->fitnessComputation();
            //			copy3->dumpPhenotype(pheno);

            int copymatched = copy3->getNumInstancesMatched();
            int copyok = copy3->getNumInstancesPosOK();
            //			printf("acc %f %f Cov %f Matched %d Pos %d Ok %d %s", acc,
            //					copy3->getAccuracy2(), copy3->getCoverage(), copymatched,
            //					copy3->getNumInstancesPos(), copyok, pheno);

            if (!(copy3->getAccuracy2() < acc)) {
                if ((origmatched - copymatched) == (origok - copyok)) {

                    //printf("Eliminado attributo %d\n", i);
                    atts++;
                    i--;
                    orig = copy3;

                    if (copy3->getAccuracy2() > acc)
                        acc = copy3->getAccuracy2();
                }

            }

        }

    }

    if(orig->getAccuracy2() != acc || orig->getCoverage() != cov) {
        printf("Class %d: Pruned %d atts Prev: Acc %f Cov %f New: Acc %f Cov %f\n",index,atts,acc,cov,orig->getAccuracy2(),orig->getCoverage());
    }
    return orig;
}

int classifier_hyperrect_list::cleanAttributes(int index) {

    int ** matched = fitnessComputationWithMatchSet(0);

    float accu = getAccuracy2();
    float cove = getCoverage();
    instance ** insIter = is->getInstancesOfIteration();

    int offset = 0;
    int change = 0;
    for (int i = 0; i < numAtt; i++) {

        int index = whichAtt[i];
        if (getAttributeSpecificity(i) != 1.0) {
            if (ai.getTypeOfAttribute(index) == NOMINAL
                    && tReal->attributeSize[index] > 2) {

                for (int i = 0; i < tReal->attributeSize[index]; i++) {

                    if (predicates[offset + i] == 1) {
                        int res = 0;
                        for (int k = 0; !res && k < matched[1][0]; k++) {
                            if (insIter[matched[0][k]]->realValues[index]
                                    == i) {
                                res = 1;
                            }
                        }

                        if (res == 0) {
                            predicates[offset + i] = res;
                            change = 1;
                        }

                    }
                }
            } else if (ai.getTypeOfAttribute(index) == REAL) {
                float min = ai.getMaxDomain(index);
                float max = ai.getMinDomain(index);
                for (int k = 0; k < matched[1][0]; k++) {
                    if (insIter[matched[0][k]]->realValues[index] > max) {
                        max = insIter[matched[0][k]]->realValues[index];
                    }
                    if (insIter[matched[0][k]]->realValues[index] < min) {
                        min = insIter[matched[0][k]]->realValues[index];
                    }

                }

                if (min > predicates[offset]) {
                    predicates[offset] = min;
                    change = 1;
                }

                if (max < predicates[offset + 1]) {
                    predicates[offset + 1] = max;
                    change = 1;
                }
            }
        }

        offset += tReal->attributeSize[index];

    }

    fitnessComputation();
    if(accu != getAccuracy2() || cove != getCoverage())
        printf("Class %d cleaned Prev: Acc %f Cov %f New: Acc %f Cov %f\n", index, accu, cove, getAccuracy2(), getCoverage());
    return change;

}

int classifier_hyperrect_list::cleanAttributes2(int index) {


    int ** matched = fitnessComputationWithMatchSet(1);

    float accu = getAccuracy2();
    float cove = getCoverage();
    instance ** insIter = is->getInstancesOfIteration();

    int offset = 0;
    int change = 0;
    for (int i = 0; i < numAtt; i++) {

        int index = whichAtt[i];
        if (getAttributeSpecificity(i) != 1.0) {
            if (ai.getTypeOfAttribute(index) == NOMINAL
                    && tReal->attributeSize[index] > 2) {

                for (int i = 0; i < tReal->attributeSize[index]; i++) {

                    if (predicates[offset + i] == 1) {
                        int resNeg = 0;
                        int resPos = 0;
                        for (int k = 0; !resNeg && k < matched[2][1]; k++) {
                            if (insIter[matched[1][k]]->realValues[index]
                                    == i) {
                                resNeg = 1;
                            }
                        }

                        for (int k = 0; !resPos && k < matched[2][0]; k++) {
                            if (insIter[matched[0][k]]->realValues[index]
                                    == i) {
                                resPos = 1;
                            }
                        }

                        if(resNeg && !resPos) {
                            predicates[offset + i] = 0;
                            change = 1;
                        }

                    }
                }
            } else if (ai.getTypeOfAttribute(index) == REAL) {

                float min = predicates[offset];
                float max = predicates[offset+ 1];

                vector<float> allPosValues;
                vector<float> allNegValues;
                for (int k=0; k < matched[2][0]; k++) {
                    allPosValues.push_back(insIter[matched[0][k]]->realValues[index]);
                }
                for (int k=0; k < matched[2][1]; k++) {
                    allNegValues.push_back(insIter[matched[1][k]]->realValues[index]);
                }

                sort(allPosValues.begin(),allPosValues.end());
                sort(allNegValues.begin(),allNegValues.end());

                min = allPosValues[0];
                max = allPosValues[allPosValues.size() -1];

                if(matched[2][1] > 0) {

                    int foundmin = 0;
                    if (allNegValues[0] >= min) foundmin =1;
                    int foundmax = 0;
                    if(allNegValues[matched[2][1]- 1] <= max) foundmax=1;

                    //printf("Negs %d Min %f Max %f foundmin %d foundmax %d\n",matched[2][1],min,max,foundmin,foundmax);
                    for (int k=0; k < (matched[2][1]) && (!foundmin || !foundmax); k++) {
                        //printf("Observing k %d-  %f %f\n",k,allNegValues[k],allNegValues[allNegValues.size() -1 -k]);
                        if (!foundmin && k > 0 && allNegValues[k] >= min) {
                            min=(allNegValues[k-1]+min)/2.0;
                            foundmin =1;
                            //printf("Changed min %f\n",min);
                        }

                        if (!foundmax && k > 0 && allNegValues[allNegValues.size() -1 -k] <= max){
                            max=(allNegValues[allNegValues.size() -1 -k + 1]+max)/2.0;
                            foundmax =1;
                            //printf("Changed max %f\n",max);
                        }
                    }
                }

                if (min > predicates[offset]) {
                    predicates[offset] = min;
                    change = 1;
                }

                if (max < predicates[offset + 1]) {
                    predicates[offset + 1] = max;
                    change = 1;
                }
            }
        }

        offset += tReal->attributeSize[index];

    }


    fitnessComputation();
    if(accu != getAccuracy2() || cove != getCoverage())
        printf("Class %d cleaned2 Prev: Acc %f Cov %f New: Acc %f Cov %f\n", index, accu, cove, getAccuracy2(), getCoverage());
    return change;

}


void classifier_hyperrect_list::finalizeEval() {
    delete listRealAtt;
    delete listDiscreteAtt;
    delete listRealPos;
    delete listDiscretePos;
}

float classifier_hyperrect_list::calculateDistance(classifier * c2) {
    classifier_hyperrect_list* c = static_cast<classifier_hyperrect_list*>(c2);

    int resultNominal = 0;
    int totalNominal = 0;
    float resultReal = 0;
    int totalReal = 0;
    int totalAtts = 0;
    int j = 0;
    for (int i = 0; i < numAtt || j < c->numAtt;) {

        totalAtts++;

        if (i >= numAtt) {
            int att = c->whichAtt[j];
            int base2 = c->offsetPredicates[j];

            if (ai.getTypeOfAttribute(att) == NOMINAL) {
                int size = tReal->attributeSize[att];
                totalNominal += size;

                for (int k = 0; k < size; k++) {

                    if (c->predicates[base2 + k] == 1) {
                        resultNominal += 1;
                    }
                }
            } else {

                totalReal++;
                resultReal += (c->predicates[base2 + 1] - c->predicates[base2])
                        / float(ai.getMaxDomain(att) - ai.getMinDomain(att));

            }

            j++;
        } else if (j >= c->numAtt) {
            int att = whichAtt[i];
            int base1 = offsetPredicates[i];
            if (ai.getTypeOfAttribute(att) == NOMINAL) {
                int size = tReal->attributeSize[att];
                totalNominal += size;
                for (int k = 0; k < size; k++) {

                    if (predicates[base1 + k] == 1) {
                        resultNominal += 1;
                    }
                }
            } else {
                totalReal++;
                resultReal += (predicates[base1 + 1] - predicates[base1])
                        / float(ai.getMaxDomain(att) - ai.getMinDomain(att));
            }
            i++;

            // The attributes are the same. Count common ones
        } else if (whichAtt[i] == c->whichAtt[j]) {
            int att = whichAtt[i];
            int base1 = offsetPredicates[i];
            int base2 = c->offsetPredicates[j];
            if (ai.getTypeOfAttribute(att) == NOMINAL) {
                int size = tReal->attributeSize[att];
                totalNominal += size;
                for (int k = 0; k < size; k++) {
                    if (predicates[base1 + k] == c->predicates[base2 + k]) {
                        resultNominal += 1;
                    }
                }
            } else {
                float lb1 = predicates[base1];
                float ub1 = predicates[base1 + 1];
                float lb2 = c->predicates[base2];
                float ub2 = c->predicates[base2 + 1];

                totalReal++;
                if (ub1 <= lb2 || lb1 >= ub2) {
                    break;
                } else {
                    float maxlower = max(lb1, lb2);
                    float minupper = min(ub1, ub2);
                    resultReal +=
                            (minupper - maxlower)
                            / float(
                                ai.getMaxDomain(att)
                                - ai.getMinDomain(att));
                }
            }

            j++;
            i++;

            //The first classifier starts in a previous attribute than the other
        } else if (whichAtt[i] < c->whichAtt[j]) {

            int att = whichAtt[i];
            int base1 = offsetPredicates[i];
            if (ai.getTypeOfAttribute(att) == NOMINAL) {
                int size = tReal->attributeSize[att];
                totalNominal += size;

                for (int k = 0; k < size; k++) {

                    if (predicates[base1 + k] == 1) {
                        resultNominal += 1;
                    }
                }
            } else {
                totalReal++;
                resultReal += (predicates[base1 + 1] - predicates[base1])
                        / float(ai.getMaxDomain(att) - ai.getMinDomain(att));
            }
            i++;
            //The second classifier represents more attributes than the first
        } else {

            int att = c->whichAtt[j];
            int base2 = c->offsetPredicates[j];
            if (ai.getTypeOfAttribute(att) == NOMINAL) {
                int size = tReal->attributeSize[att];
                totalNominal += size;

                for (int k = 0; k < size; k++) {

                    if (c->predicates[base2 + k] == 1) {
                        resultNominal += 1;
                    }
                }
            } else {
                totalReal++;
                resultReal += (c->predicates[base2 + 1] - c->predicates[base2])
                        / float(ai.getMaxDomain(att) - ai.getMinDomain(att));
            }

            j++;
        }

    }

    float nominalTerm = 0;
    float realTerm = 0;

    if (totalNominal > 0) {
        nominalTerm = ((float) resultNominal / (float) totalNominal);
    }

    if (totalReal > 0) {
        realTerm = float(resultReal);
    }

    int missingAtts=ai.getNumAttributesMC()-totalAtts;

    float final = (nominalTerm * (totalAtts - totalReal) / totalAtts)
            + (realTerm * totalReal / totalAtts);

    final = final*totalAtts/(float)ai.getNumAttributesMC() + missingAtts/(float)ai.getNumAttributesMC();

    return final;
}



