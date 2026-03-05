#include "instanceSet.h"
#include "configManagement.h"
#include "instance.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "JVector.h"
#include "JString.h"
#include "attributesInfo.h"
#include "utils.h"
#include "windowingILAS.h"
#include "windowingGWS.h"
#include "random.h"
#include "messageBuffer.h"
#include "timeManagement.h"
#include "timerMDL.h"


extern messageBuffer mb;
extern attributesInfo ai;
extern Random rnd;
extern configManagement cm;
extern timeManagement tm;
extern int nodeRank;
extern timerMDL *tMDL;

void instanceSet::readFile(char fileName[], int &numInstances, int traintest)
{
	FILE *fp;
	char string[200000];
	JVector<instance *> tempSet(1000,100000);
	int num = 0;

	fp = fopen(fileName, "r");
	if (!fp) {
		fprintf(stderr,"Can't open %s\n",fileName);
		exit(1);
	}

	parseHeader(fp, traintest);

	fgets(string, 199999, fp);
	while (!feof(fp)) {
		string[strlen(string) - 1] = 0;
		if(string[strlen(string) - 1]==13) 
			string[strlen(string) - 1] = 0;
		if (string[0] != '%' && strlen(string) > 0) {
			instance *tmp=new instance(num,string,traintest);
			if(traintest==TRAIN) ai.insertInstance(tmp);
			tempSet.addElement(tmp);
			num++;
		}
		fgets(string, 199999, fp);
	}
	fclose(fp);

	numInstances = num;
	set=new instance*[num];
	for(int i=0;i<num;i++) {
		set[i]=tempSet[i];
	}
	tempSet.removeAllElements();
	tempSet.trimToSize();
	if(traintest==TRAIN) ai.calculateAverages();
}

void instanceSet::parseRelation(char *string)
{
	char cadena[5000];

	string += 9;
	while (*string == ' ' || *string == '\t')
		string++;
	mb.printf("Dataset name: %s\n", string);
}

void instanceSet::parseReal(int numAttr)
{
	ai.setTypeOfAttribute(numAttr,REAL);
	mb.printf("Attribute %d real valued\n", numAttr);
}

void instanceSet::parseInteger(char *string, int numAttr)
{
/*	int min, max;
	if (sscanf(string+8, "[%d,%d]", &min, &max) != 2) {
		fprintf(stderr,"Parse error:%s\n", string+8);
		exit(1);
	}
	if (min >= max) {
		fprintf(stderr,"Attribute %d inconsistent:%d %d\n",
		       numAttr, min, max);
		exit(1);
	}
	ai.setTypeOfAttribute(numAttr,REAL);
	mb.printf("Attribute %d integer [%d:%d]\n", numAttr, min, max);*/
	ai.setTypeOfAttribute(numAttr,REAL);
	mb.printf("Attribute %d integer\n", numAttr);

}

void instanceSet::parseNominal(char *string, int numAttr)
{
	mb.printf("Attribute %d nominal\n", numAttr);
	ai.setTypeOfAttribute(numAttr,NOMINAL);
	int numValues = 0;
	char *ptr = string;
	ptr++;
	while (*ptr != '}') {
		char value[500];
		int size;
		while (*ptr == ' ' || *ptr == '\t') ptr++;
		for (size = 0;
		     !(*ptr == 0 || *ptr == ',' || *ptr == '}'); 
		     value[size++] = *ptr++);
		if (*ptr == 0) {
			fprintf(stderr,"Parse error: %s\n", string);
			exit(1);
		}
		value[size] = 0;
		if (size > 0) {
			JString *str = new JString(value);
			ai.insertNominalValue(numAttr,str);
			mb.printf("Value %d of attribute %d: %s\n", numValues,
			       numAttr, value);
			numValues++;
		}
		if (*ptr == ',') ptr++;
	}
}

void instanceSet::parseAttribute(char *string, int numAttr)
{
	char name[5000];

	string += 10;
	while (*string == ' ' || *string == '\t')
		string++;

	if(*string=='\'') {
		string++;
		int count=0;
		while(*string!='\'') {
			name[count++]=*string++;
		}
		name[count]=0;
		string++;
	} else {
		if (sscanf(string, "%s", name) != 1) {
			fprintf(stderr,"Parse error:%s\n", string);
			exit(1);
		}
		string += strlen(name);
	}


	while (*string == ' ' || *string == '\t')
		string++;

	mb.printf("Attribute %d:Name %s Def:%s\n", numAttr, name, string);

	ai.insertAttributeName(new JString(name));

	if (!strcasecmp(string, "real") || !strcasecmp(string, "numeric")) {
		parseReal(numAttr);
	} else if (!strncasecmp(string, "integer", 7)) {
		parseInteger(string, numAttr);
	} else if (string[0] == '{') {
		parseNominal(string, numAttr);
	} else {
		fprintf(stderr,"Unknown attribute type %s\n", string);
		exit(1);
	}
}

void instanceSet::parseHeader(FILE * fp,int traintest)
{
	JVector<char *>header;

	char string[10000];
	int end = 0;
	int numAttr = 0;

	fgets(string, 9999, fp);
	while (!feof(fp) && !end) {
		string[strlen(string)-1]=0;
		if(string[strlen(string) - 1]==13) 
			string[strlen(string) - 1] = 0;
		if (string[0] != '%' && strlen(string) > 1) {
			if (!strncasecmp(string, "@relation", 9)) {
				if (traintest == TRAIN) parseRelation(string);
			} else if (!strncasecmp(string, "@attribute", 10)) {
				if (traintest == TRAIN) {
					char *tmp=new char[strlen(string)+1];
					strcpy(tmp,string);
					header.addElement(tmp);
					numAttr++;
				}
			} else if (!strncasecmp(string, "@data", 5)) {
				end = 1;
			} else {
				fprintf(stderr,"Unknown header element:%s\n"
					, string);
				exit(1);
			}
		}
		if(!end) 
			fgets(string, 9999, fp);
	}

	if (traintest == TRAIN) {
		ai.setNumAttributes(numAttr);
		for(int i=0;i<numAttr;i++) {
			parseAttribute(header[i], i);
			delete header[i];
		}
		if (ai.getTypeOfAttribute(numAttr - 1) != NOMINAL) {
			fprintf(stderr,"Class attribute (last) should be nominal\n");
			exit(1);
		}
	}
}

void instanceSet::initializeWindowing(int traintest)
{
	if(traintest == TRAIN) {
		if(cm.thereIsParameter(WINDOWING_ILAS)) {
			windowingEnabled=1;
			if(win==NULL) win=new windowingILAS;
		} else if(cm.thereIsParameter(WINDOWING_GWS)) {
			windowingEnabled=1;
			if(win==NULL) win=new windowingGWS;
		} else {
			windowingEnabled=0;
			window=NULL;
		}

		if(windowingEnabled) {
			win->setInstances(set,numInstances);
			win->newIteration(window,windowSize,strataOffset);
		}
	} else {
		windowingEnabled=0;
		window=NULL;
	}
}

instanceSet::instanceSet(char fileName[], int traintest)
{
	int i;

	window=NULL;
	win=NULL;
	readFile(fileName, numInstances, traintest);
	if (!numInstances) {
		fprintf(stderr,"Instances file %s is empty\n",fileName);
		exit(1);
	}

	if(!cm.thereIsParameter(IGNORE_MISSING_VALUES)) {
		for(i=0;i<numInstances;i++) {
			set[i]->updateMissing();
		}
	}
	
	if(cm.thereIsParameter(ROTATE_HYPERRECTANGLES)) {
		for(i=0;i<numInstances;i++) {
			set[i]->normalize();
		}
	}

	origSet = new instance *[numInstances];
	numInstancesOrig=numInstances;
	for(i=0;i<numInstances;i++) {
		origSet[i]=set[i];
	}

	initializeWindowing(traintest);

	if(traintest==TRAIN) {
		numClasses=ai.getNumClasses();
		classWiseInit=cm.thereIsParameter(CLASS_WISE_INIT);
		initInstanceLists();
	} else {
		initSamplings=NULL;
		countsByClass=NULL;
		instByClass=NULL;
	}
}

instanceSet::~instanceSet()
{
	int i;

	for (i = 0; i < numInstancesOrig; i++) delete origSet[i];
	delete origSet;
	delete set;

	if(initSamplings) {
		for(i=0;i<numClasses;i++) {
			delete initSamplings[i];
			delete instByClass[i];
		}
		delete initSamplings;
		delete instByClass;
		delete countsByClass;
	}

	if(win) delete win;
}

int instanceSet::getNumInstancesOfIteration()
{
	if(windowingEnabled) return windowSize;
	return numInstances;
}

int instanceSet::getStrataOffsetOfIteration()
{
	if(windowingEnabled) return strataOffset;
	return 0;
}

int instanceSet::newIteration(int isLast)
{
	if(isLast) {
		//mb.printf("Windowing disabled for last iteration\n");
		windowingEnabled=0;
		return 1;
	}

	if(windowingEnabled) {
		win->newIteration(window,windowSize, strataOffset);
		if(win->needReEval()) return 1;
		return 0;
	}

	return 0;
}

void instanceSet::initInstanceLists()
{
	int i;
	int numInst=getNumInstances();

	countsByClass = new int[numClasses];
	initSamplings = new Sampling *[numClasses];
	instByClass = new int *[numClasses];

	for(i=0;i<numClasses;i++) {
		countsByClass[i]=0;
	}

	for(i=0;i<numInst;i++) {
		int cl=set[i]->getClass();
		countsByClass[cl]++;
	}

	for(i=0;i<numClasses;i++) {
		int num=countsByClass[i];
		initSamplings[i] = new Sampling(num);
		instByClass[i] = new int[num];
		countsByClass[i]=0;
	}

	for(i=0;i<numInst;i++) {
		int cl=set[i]->getClass();
		instByClass[cl][countsByClass[cl]++]=i;
	}
}

instance *instanceSet::getInstanceInit(int forbiddenCL) 
{
	if(classWiseInit) {
		if(forbiddenCL!=numClasses) {
			int allEmpty=1;
			int i;

			for(i=0;i<numClasses;i++) {
				if(i==forbiddenCL) continue;
				if(countsByClass[i]>0) {
					allEmpty=0;
					break;
				}
			}
			if(allEmpty) {
				return NULL;
			}
		}

		int cl;
		do {
			if(forbiddenCL!=numClasses) {
				cl=rnd(0,numClasses-2);
				if(cl>=forbiddenCL) cl++;
			} else {
				cl=rnd(0,numClasses-1);
			}
		} while(countsByClass[cl]==0);

		int pos=initSamplings[cl]->getSample();
		int insIndex=instByClass[cl][pos];
		instance *ins=set[insIndex];

		return ins;
	} else {
		int nc=numClasses;
		int count[nc];
		int total=0;
		int i;
		if(forbiddenCL!=numClasses) nc--;
	
		for(i=0;i<nc;i++) {
			if(i<forbiddenCL) 
				count[i]=initSamplings[i]->numSamplesLeft();
			else 
				count[i]=initSamplings[i+1]->numSamplesLeft();
			total+=count[i];
		}
		int pos=rnd(0,total-1);
		int acum=0;
		int found=0;
		for(i=0;i<nc && !found;i++) {
			acum+=count[i];
			if(pos<acum) {
				found=1;
			}
		}
		i--;
		if(i>=forbiddenCL) i++;

		pos=initSamplings[i]->getSample();
		int insIndex=instByClass[i][pos];
		instance *ins=set[insIndex];

		return ins;
	}
}


void instanceSet::removeInstancesAndRestart(classifier *cla)
{
	int i;

	if(initSamplings) {
		for(i=0;i<numClasses;i++) {
			delete initSamplings[i];
			delete instByClass[i];
		}
		delete initSamplings;
		delete instByClass;
		delete countsByClass;
	}

	int numRemoved=0;
	int index=0;
	int countClassRem[numClasses];

	for(i=0;i<numClasses;i++) {
		countClassRem[i]=0;
	}

	int numOrig=numInstances;

	cla->initiateEval();
	while(index<numInstances) {
		if(cla->doMatch(set[index])) {
			//delete set[index];
			set[index]=set[numInstances-1];
			numRemoved++;
			numInstances--;
		} else {
			countClassRem[set[index]->instanceClass]++;
			index++;
		}
	}
	cla->finalizeEval();

	ai.updateClassCounters(countClassRem);

	/*if(tMDL->mdlAccuracy) {
		for(i=0;i<numClasses;i++) {
			double ratio=((double)ai.getInstancesOfClass(i) / (double)countClassRem[i])
				* tMDL->origCoverageBreaks[i];
			tMDL->coverageBreaks[i]=ratio;
			if(tMDL->coverageBreaks[i]>1) {
				tMDL->coverageBreaks[i]=1;
			}
			if(nodeRank==0) {
				mb.printf("New coverage break for class %d:%f\n",i,
					tMDL->coverageBreaks[i]);
			}
		}
	}*/

	/*if(cm.thereIsParameter(WINDOWING_ILAS)) {
		double ratio=(double)numOrig/(double)numInstances;
		double numS=cm.getParameter(WINDOWING_ILAS)/ratio;
		if(numS<2) {
			numS=2;
		}
		if(nodeRank==0) {
			mb.printf("New number of strata:%f\n",numS);
		}
		cm.setParameter(numS,WINDOWING_ILAS);
	}*/
	

	if(nodeRank==0) {
            mb.printf("Removed %d instances. %d Instances left. Acc of rule %f-%f \n"
                        ,numRemoved,numInstances,cla->getAccuracy(),cla->getAccuracy2());
	}


	initInstanceLists();

	initializeWindowing(TRAIN);
}

void instanceSet::restart()
{
	int i;

	if(initSamplings) {
		for(i=0;i<numClasses;i++) {
			delete initSamplings[i];
			delete instByClass[i];
		}
		delete initSamplings;
		delete instByClass;
		delete countsByClass;
	}

	initInstanceLists();

	initializeWindowing(TRAIN);
}

int instanceSet::getMajorityClass()
{
	int counters[numClasses];
	int i;

	for(i=0;i<numClasses;i++) counters[i]=0;

	for(i=0;i<numInstances;i++) {
		counters[set[i]->getClass()]++;
	}

	int max=counters[0];
	int classMax=0;
	for(i=1;i<numClasses;i++) {
		if(counters[i]>max) {
			max=counters[i];
			classMax=i;
		}
	}

	return classMax;
}

int instanceSet::getMajorityClassExcept(int cl)
{
	int counters[numClasses];
	int i;

	for(i=0;i<numClasses;i++) counters[i]=0;

	for(i=0;i<numInstances;i++) {
		int insCl=set[i]->getClass();
		if(insCl!=cl) counters[insCl]++;
	}

	int max=counters[0];
	int classMax=0;
	for(i=1;i<numClasses;i++) {
		if(counters[i]>max) {
			max=counters[i];
			classMax=i;
		}
	}

	return classMax;
}
