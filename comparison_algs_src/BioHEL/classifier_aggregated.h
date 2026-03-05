#ifndef _CLASSIFIER_AGGREGATED_

#define _CLASSIFIER_AGGREGATED_

#include "classifier.h"
#include "JVector.h"
#include "instanceSet.h"
#include "configManagement.h"

extern configManagement cm;

class classifier_aggregated {
	JVector<classifier *>classifiers;
	int defaultClass;
	double accuracy;
	int defaultClassPolicy;

public:
	inline classifier_aggregated() {
		defaultClassPolicy=(int)cm.getParameter(DEFAULT_CLASS);
                switch(defaultClassPolicy) {
                case MAJOR:
                    defaultClass=ai.getMostFrequentClass();
                    //defaultClass=0;
                    break;
                case MINOR:
                    //defaultClass=0;
                    defaultClass=ai.getLeastFrequentClass();
                    break;
                case FIXED:
                    defaultClass=(int)cm.getParameter(FIXED_DEFAULT_CLASS);
                    break;
                case DISABLED:
                default:
                    defaultClass=-1;
                    break;
                }
	}

	inline ~classifier_aggregated() {
		int i;

		for(i=0;i<classifiers.size();i++) delete classifiers[i];
	}

	inline int getClass(int classifier) {
		if(defaultClass!=-1 && classifier==classifiers.size()) 
			return defaultClass;
		return classifiers[classifier]->getClass();
	}

	inline int getNumClassifiers() {
		int numCL=classifiers.size();
		if(defaultClass!=-1) numCL++;
		return numCL;
	}

	inline void setDefaultRule(instanceSet *is) {
		int i;

		if(defaultClassPolicy!=DISABLED) return;

		int nc=ai.getNumClasses();
		int classCounts[nc];
		for(i=0;i<nc;i++) classCounts[i]=0;

		int numInst=is->getNumInstances();
		instance **instances=is->getAllInstances();
		for(i=0;i<numInst;i++) {
			classCounts[instances[i]->instanceClass]++;
		}

		int max=classCounts[0];
		int posMax=0;
		for(i=1;i<nc;i++) {
			if(classCounts[i]>max) {
				posMax=i;
				max=classCounts[i];
			}
		}

		defaultClass=posMax;
	}

	inline int classify(instance *ins) {
		int i;

		int size=classifiers.size();
		for(i=0;i<size;i++) {
			if(classifiers[i]->doMatch(ins)) return i;
		}
		if(defaultClass!=-1) return size;
		return -1;
	}

	inline double getAccuracy() {return accuracy;}
	inline void setAccuracy(double acc) {accuracy=acc;}

	inline void dumpPhenotype(char *string) {
		int i;
		char temp[2000000];
		
		int size=classifiers.size();
		strcpy(string,"");
		for(i=0;i<size;i++) {
			sprintf(temp,"%d:",i);
			strcat(string,temp);
			classifiers[i]->dumpPhenotype(temp);
			strcat(string,temp);
		}
		if(defaultClass!=-1) {
			sprintf(temp, "%d:Default rule -> %s\n", i
				,ai.getNominalValue(tGlobals->numAttributesMC
					,defaultClass)->cstr());
			strcat(string, temp);
		}
		strcat(string, "\n");
	}

	inline void addClassifier(classifier *cl) {
		cl->initiateEval();
		classifiers.addElement(cl);
	}
};

#endif

