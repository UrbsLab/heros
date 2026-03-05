#include "agentPerformance.h"
#include "timerGlobals.h"
#include "timerHierar.h"
#include "timerMDL.h"
#include "classifier.h"
#include "messageBuffer.h"

extern timerGlobals *tGlobals;
extern timerMDL *tMDL;
extern timerHierar *tHierar;
extern messageBuffer mb;

agentPerformance::agentPerformance(int pNumClassifiers,int pNumClasses)
{
	numClasses=pNumClasses;
	numClassifiers=pNumClassifiers;

	numInstancesOK = 0;
	numInstancesKO = 0;
	numInstancesNC = 0;
	numInstancesTotal = 0;
	aliveClassifiers=0;

	int i, j;
	statisticsForEachClass = new int *[numClasses];
	statisticsConfusionMatrix = new int *[numClasses];
	for (i = 0; i < numClasses; i++) {
		statisticsConfusionMatrix[i] = new int[numClasses];
		statisticsForEachClass[i] = new int[3];
	}


	classifierActivated = new int[numClassifiers];
	classifierCorrect = new int[numClassifiers];
	classifierWrong = new int[numClassifiers];
	for (i = 0; i < numClassifiers; i++) {
		classifierActivated[i] = 0;
		classifierCorrect[i] = 0;
		classifierWrong[i] = 0;
	}

	for (i = 0; i < numClasses; i++) {
		for (j = 0; j < numClasses; j++)
			statisticsConfusionMatrix[i][j] = 0;
		for (j = 0; j < 3; j++)
			statisticsForEachClass[i][j] = 0;
	}
}

void agentPerformance::addPrediction(int realClass,int predictedClass
	,int usedClassifier)
{
	numInstancesTotal++;
	if(usedClassifier!=-1) {
		if(!classifierActivated[usedClassifier]) {
			aliveClassifiers++;
		}
		classifierActivated[usedClassifier]++;
		statisticsConfusionMatrix[realClass][predictedClass]++;
		if (predictedClass == realClass) {
			numInstancesOK++;
			statisticsForEachClass[realClass][0]++;
			classifierCorrect[usedClassifier]++;
		} else {
			classifierWrong[usedClassifier]++;
			numInstancesKO++;
			statisticsForEachClass[realClass][1]++;
		}
	} else {
		numInstancesNC++;
		statisticsForEachClass[realClass][2]++;
	}
}

void agentPerformance::dumpStatsBrief()
{
	int i,j;

        mb.printf("Accuracy : %f\n", numInstancesOK / numInstancesTotal);
        for (i = 0; i < numClasses; i++) mb.printf("%d\t", i);
        mb.printf("\n");
        for (i = 0; i < numClasses; i++) {
                for (j = 0; j < numClasses; j++)
                        mb.printf("%d\t", statisticsConfusionMatrix[i][j]);
                mb.printf("\n");
        }
}

void agentPerformance::dumpStats2()
{
	int i;
        for (i = 0; i < numClassifiers; i++)
                mb.printf("Accuracy of rule %d:%f/%d\n", i, 
		       (double)classifierCorrect[i]/(double)classifierActivated[i]
                       ,classifierActivated[i]);
}


void agentPerformance::dumpStats(const char *prefix)
{
	int i,j;

        mb.printf("%s accuracy : %f\n", prefix,
               numInstancesOK / numInstancesTotal);

        mb.printf("%s error : %f\n", prefix,
               numInstancesKO / numInstancesTotal);
        mb.printf("%s not classified : %f\n", prefix,
               numInstancesNC / numInstancesTotal);
        mb.printf("%s For each class:\n", prefix);
        for (i = 0; i < numClasses; i++) {
                mb.printf("%d: accuracy : %d\n", i, statisticsForEachClass[i][0]);
                mb.printf("%d: error : %d\n", i, statisticsForEachClass[i][1]);
                mb.printf("%d: not classified : %d\n", i,
                       statisticsForEachClass[i][2]);
        }

	mb.printf("%s Confusion Matrix. Row real class, Column predicted class\n"
		, prefix);
        for (i = 0; i < numClasses; i++) mb.printf("%d\t", i);
        mb.printf("\n");
        for (i = 0; i < numClasses; i++) {
                for (j = 0; j < numClasses; j++)
                        mb.printf("%d\t", statisticsConfusionMatrix[i][j]);
                mb.printf("\n");
        }

        mb.printf("Performance of each classifier:\n");
        for (i = 0; i < numClassifiers; i++)
                mb.printf("Classifier %d: %d/%d=%f%c\n", i, classifierCorrect[i],
                       classifierActivated[i], (double) classifierCorrect[i] /
                       (double) classifierActivated[i] * 100.0, '%');
}

agentPerformance::~agentPerformance()
{
	delete classifierActivated;
	delete classifierCorrect;
	delete classifierWrong;

	int i;
	for (i = 0; i < numClasses; i++) {
                delete statisticsConfusionMatrix[i];
	}
        delete statisticsConfusionMatrix;

        for (i = 0; i < numClasses; i++) {
                delete statisticsForEachClass[i];
	}
        delete statisticsForEachClass;
}

double agentPerformance::getAverageActivation()
{
	/*int num=numClassifiers;
	int i;
	double count=0;
	
	if(tGlobals->defaultClassPolicy!=DISABLED) num--;
	for(i=0;i<num;i++) {
		if(classifierActivated[i]>=3) count++;
	}
	//count/=(double)num;

	return count;*/
	double actDR=0;
	if(tGlobals->defaultClassPolicy!=DISABLED) 
		actDR=classifierActivated[numClassifiers-1];
	return (numInstancesTotal-numInstancesNC-actDR)/numInstancesTotal;
}

