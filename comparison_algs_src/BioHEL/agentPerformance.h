#ifndef _AGENT_PERFORMANCE_H_
#define _AGENT_PERFORMANCE_H_

class classifier;

class agentPerformance {
	int numClassifiers;
	int numClasses;

	double numInstancesOK;
	double numInstancesKO;
	double numInstancesNC;
	double numInstancesTotal;
	int **statisticsForEachClass;
	int **statisticsConfusionMatrix;
	int *classifierActivated;
	int *classifierCorrect;
	int *classifierWrong;
	int aliveClassifiers;

public:	
	agentPerformance(int pNumClassifiers,int pNumClasses);
	~agentPerformance();
	void addPrediction(int realClass,int predictedClass,int usedClassifier);
	inline double getAccuracy() { return numInstancesOK/numInstancesTotal; }
	inline double getError(){return numInstancesKO/numInstancesTotal;}
	inline double getNC(){return numInstancesNC/numInstancesTotal;}
	inline int getNumError(){return (int)numInstancesKO;}
	inline int getNumNC(){return (int)numInstancesNC;}
	inline int getActivationsOfClassifier(int classifier) { 
		return classifierActivated[classifier]; 
	}
	int getCorrectPredictionsOfClassifier(int classifier) {
		return classifierCorrect[classifier];
	}
	double getAccOfClassifier(int classifier) {
		return (double)classifierCorrect[classifier]
			/(double)classifierActivated[classifier];
	}
	void dumpStats(const char *prefix);
	void dumpStats2();
	void dumpStatsBrief();
	int getAliveClassifiers(){return aliveClassifiers;}
	double getAverageActivation();
	void disableClassifier(int classifier) {
		classifierActivated[classifier]=0;
	}

	double getLSacc(int classifier) {
		if(classifierActivated[classifier]==0) return 0;
		double acc=getAccOfClassifier(classifier);
		double laplaceAcc=(classifierCorrect[classifier]+1.0)
			/(classifierActivated[classifier]+numClasses);
		return (acc<laplaceAcc?acc:laplaceAcc);
	}

	int isClassifierWrong(int classifier) {
		return classifierWrong[classifier];
	}

};

#endif
