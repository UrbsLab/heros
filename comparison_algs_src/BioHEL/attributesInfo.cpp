#include "attributesInfo.h"
#include <stdio.h>
#include "messageBuffer.h"
#include "configManagement.h"
#include <math.h>

extern messageBuffer mb;
extern configManagement cm;

attributesInfo::attributesInfo()
{
	numAttributes=-1;
	numExamples=0;
	thereAreNominal=thereAreRealValued=0;
}

void attributesInfo::setNumAttributes(int num)
{
	int i;

	numAttributes=num;
	numAttributesMC=num-1;
	typeOfAttributes=new int[num];
	valuesOfNominalAttributes=new JVector<JString *>[num];

	minDomain=new float[num-1];
	maxDomain=new float[num-1];
	sizeDomain=new float[num-1];
	sizeDomain2=new float[num-1];

	averageOfAttribute=new float *[num-1];
	deviationOfAttribute=new float *[num-1];
	countNumValuesForRealAttributes=new int *[num-1];	
	valueFrequenciesForNominalAttributes=new float **[num-1];
	mostFrequentValueForNominalAttributes=new int *[num-1];

	for(i=0;i<num;i++) {
		typeOfAttributes[i]=-1;
	}
}


void attributesInfo::setTypeOfAttribute(int attribute,int type)
{
#ifdef DEBUG
	if(attribute<0 || attribute>=numAttributes ) {
		fprintf(stderr,"Incorrect values at attributesInfo::setTypeOfAttribute %d\n",attribute);
		exit(1);
	}

	if(typeOfAttributes[attribute]!=-1) {
		fprintf(stderr,"Already defined type for attribute %d\n",attribute);
		exit(1);
	}
#endif

	if(attribute<numAttributesMC) {
		if(type==REAL) thereAreRealValued=1;
		if(type==NOMINAL) thereAreNominal=1;
	}
	typeOfAttributes[attribute]=type;
}


void attributesInfo::insertNominalValue(int attribute,JString *value)
{
#ifdef DEBUG
	if(attribute<0 || attribute>=numAttributes) {
		fprintf(stderr,"Incorrect value at attributesInfo::insertNominalValue %d\n",attribute);
		exit(1);
	}

	if(typeOfAttributes[attribute]!=NOMINAL) {
		fprintf(stderr,"Attribute %d is not nominal at attributesInfo::insertNominalValue %d\n",attribute);
		exit(1);
	}
#endif

	valuesOfNominalAttributes[attribute].addElement(value);
}

JString *attributesInfo::getNominalValue(int attribute,int value)
{
#ifdef DEBUG
	if(attribute<0 || attribute>=numAttributes) {
		fprintf(stderr,"Incorrect attr at attributesInfo::getNominalValue %d\n",attribute);
		exit(1);
	}
	if(typeOfAttributes[attribute]!=NOMINAL) {
		fprintf(stderr,"Attribute %d is not nominal at attributesInfo::insertNominalValue %d\n",attribute);
		exit(1);
	}
	if(value<0 || value>=valuesOfNominalAttributes[attribute].size()) {
		fprintf(stderr,"Incorrect value at attributesInfo::getNominalValue %d\n",value);
		exit(1);
	}
#endif


	return valuesOfNominalAttributes[attribute].elementAt(value);
}


int attributesInfo::getNumValuesAttribute(int attribute)
{
#ifdef DEBUG
	if(attribute<0 || attribute>=numAttributes) {
		fprintf(stderr,"Incorrect value at attributesInfo::getNumValuesAttribute %d\n",attribute);
		exit(1);
	}

	if(typeOfAttributes[attribute]!=NOMINAL) {
		fprintf(stderr,"Attribute %d is not nominal at attributesInfo::getNumValuesAttribute\n",attribute);
		exit(1);
	}
#endif

	return valuesOfNominalAttributes[attribute].size();
}

void attributesInfo::insertInstance(instance *ins)
{
	int i,j,k;

	//First sample
	if(!numExamples) {
		int numClasses=getNumValuesAttribute(numAttributes-1);
		classOfInstances=new int[numClasses];
		for(i=0;i<numClasses;i++) classOfInstances[i]=0;

		for(i=0;i<numAttributes-1;i++) {
			if(typeOfAttributes[i]==NOMINAL) {
				mostFrequentValueForNominalAttributes[i] = new int[numClasses];
				valueFrequenciesForNominalAttributes[i] = new float*[numClasses];
				for(j=0;j<numClasses;j++) {
					valueFrequenciesForNominalAttributes[i][j]=
						new float[getNumValuesAttribute(i)];
					for(k=0;k<getNumValuesAttribute(i);k++)
						valueFrequenciesForNominalAttributes[i][j][k]=0;
				}
			} else {
				countNumValuesForRealAttributes[i] = new int[numClasses];
				averageOfAttribute[i] = new float[numClasses];
				deviationOfAttribute[i] = new float[numClasses];
				for(j=0;j<numClasses;j++) {
					countNumValuesForRealAttributes[i][j]=0;
					averageOfAttribute[i][j]=0;
					deviationOfAttribute[i][j]=0;
				}

			}
		}
	}

	int instanceClass=ins->getClass();
	classOfInstances[instanceClass]++;

	if(ins->hasMissingValues()) {
		for(i=0;i<numAttributes-1;i++) {
			if(!ins->isMissing(i)) {
				if(typeOfAttributes[i]==NOMINAL) {
					valueFrequenciesForNominalAttributes
						[i][instanceClass]
						[ins->valueOfAttribute(i)]++;
				} else {
					float value=ins->realValueOfAttribute(i);
					averageOfAttribute[i][instanceClass]+=value;
					deviationOfAttribute[i][instanceClass]+=(value*value);
					countNumValuesForRealAttributes[i][instanceClass]++;
					if(!numExamples) {
						minDomain[i] = maxDomain[i] = value;
					} else {
						if(value<minDomain[i]) 
							minDomain[i]=value;
						if(value>maxDomain[i]) 
							maxDomain[i]=value;
					}
					sizeDomain[i]=maxDomain[i]-minDomain[i];
					sizeDomain2[i]=sizeDomain[i]/2;
				}
			}
		}
	} else {
		for(i=0;i<numAttributes-1;i++) {
			if(typeOfAttributes[i]==NOMINAL) {
				valueFrequenciesForNominalAttributes
					[i][instanceClass]
					[ins->valueOfAttribute(i)]++;
			} else {
				float value=ins->realValueOfAttribute(i);
				averageOfAttribute[i][instanceClass]+=value;
				deviationOfAttribute[i][instanceClass]+=(value*value);
				countNumValuesForRealAttributes[i][instanceClass]++;
				if(!numExamples) {
					minDomain[i] = maxDomain[i] = value;
				} else {
					if(value<minDomain[i]) 
						minDomain[i]=value;
					if(value>maxDomain[i]) 
						maxDomain[i]=value;
				}
				sizeDomain[i]=maxDomain[i]-minDomain[i];
				sizeDomain2[i]=sizeDomain[i]/2;
			}
		}
	}

	numExamples++;
}

void attributesInfo::calculateAverages() 
{
	int i,j,k;

	int numClasses=getNumValuesAttribute(numAttributes-1);

	mostFrequentClass=leastFrequentClass=0;
	int numMostFrequent=classOfInstances[0];
	int numLeastFrequent=classOfInstances[0];
	for(i=1;i<numClasses;i++) {
		if(classOfInstances[i]>numMostFrequent) {
			mostFrequentClass=i;
			numMostFrequent=classOfInstances[i];
		}
		if(classOfInstances[i]<numLeastFrequent) {
			leastFrequentClass=i;
			numLeastFrequent=classOfInstances[i];
		}
	}
	mb.printf("Least frequent class is %d\n",leastFrequentClass);
	mb.printf("Most frequent class is %d\n",mostFrequentClass);

	for(i=0;i<numAttributes-1;i++) {
		if(typeOfAttributes[i]==NOMINAL) {
			int *globalFreq=new int[getNumValuesAttribute(i)];
			for(j=0;j<getNumValuesAttribute(i);j++) globalFreq[j]=0;
			for(j=0;j<numClasses;j++) {
				for(k=0;k<getNumValuesAttribute(i);k++) {
					globalFreq[k] += (int)valueFrequenciesForNominalAttributes[i][j][k];
				}
			}
			int maxGlob=globalFreq[0];
			int bestValGlob=0;
			for(j=1;j<getNumValuesAttribute(i);j++) {
				if(globalFreq[j]>maxGlob) {
					maxGlob=globalFreq[j];
					bestValGlob=j;
				}
			}

			for(j=0;j<numClasses;j++) {
				float valuesCount
					= valueFrequenciesForNominalAttributes[i][j][0];
				int max=(int)valueFrequenciesForNominalAttributes[i][j][0];
				int bestVal=0;
				for(k=1;k<getNumValuesAttribute(i);k++) {
					valuesCount += valueFrequenciesForNominalAttributes[i][j][k];
					if(valueFrequenciesForNominalAttributes[i][j][k]>max) {
						max=(int)valueFrequenciesForNominalAttributes[i][j][k];
						bestVal=k;
					}
				}
				if(max>0) {
					mostFrequentValueForNominalAttributes[i][j]=bestVal;
				} else {
					mostFrequentValueForNominalAttributes[i][j]=bestValGlob;
				}
				//printf("Moda de Attribute %d per Classe %d:%d\n",
				//	i,j,mostFrequentValueForNominalAttributes[i][j]);
				for(k=0;k<getNumValuesAttribute(i);k++) {
					if(valuesCount)
						valueFrequenciesForNominalAttributes[i][j][k]/=valuesCount;
					else
						valueFrequenciesForNominalAttributes[i][j][k]=0;
					//printf("Frequencia de Valor %d del atribut %d per la instanceClass %d:%f\n",k,i,j,valueFrequenciesForNominalAttributes[i][j][k]);
				}
			}
		} else {
			float globalAvg=0;
			float globalDev=0;
			for(j=0;j<numClasses;j++) 
				globalAvg+=averageOfAttribute[i][j];
			for(j=0;j<numClasses;j++) 
				globalDev+=deviationOfAttribute[i][j];

			globalDev-=((globalAvg*globalAvg)/(float)numExamples);
			globalDev/=(float)(numExamples-1);
			globalDev=sqrt(globalDev);
			globalAvg/=(float)numExamples;

			for(j=0;j<numClasses;j++) {
				if(countNumValuesForRealAttributes[i][j]==0) {
					averageOfAttribute[i][j]=globalAvg;
					deviationOfAttribute[i][j]=globalDev;
				} else {
					deviationOfAttribute[i][j] -= 
						((averageOfAttribute[i][j] *
						averageOfAttribute[i][j])/(float)
						countNumValuesForRealAttributes[i][j]);
					deviationOfAttribute[i][j] /= (float)
						(countNumValuesForRealAttributes[i][j]-1);
					deviationOfAttribute[i][j] =
						sqrt(deviationOfAttribute[i][j]);
					averageOfAttribute[i][j] /= (float)
						countNumValuesForRealAttributes[i][j];
				}
				//printf("Attribute %d i Classe %d.Avg:%f Dev:%f\n"
				//	,i,j,averageOfAttribute[i][j]
				//	,deviationOfAttribute[i][j]);
			}
			//printf("Attribute %d. Min:%f, Max:%f Avg:%f Dev:%f\n"
			//	,i,minDomain[i],maxDomain[i],globalAvg
			//	,globalDev);
		}
	}

	/*if(!thereAreRealValued) {
		cm.removeParameter(KR_ADI);
		cm.removeParameter(KR_HYPERRECT);
		cm.removeParameter(KR_INSTANCE_SET);
	}*/
}

float attributesInfo::getDeviationOfAttribute(int whichClass,int attribute)
{
#ifdef DEBUG
	if(attribute<0 || attribute>=numAttributes-1) {
		fprintf(stderr,"Incorrect value at attributesInfo::getNumValuesAttribute %d\n",attribute);
		exit(1);
	}

	if(typeOfAttributes[attribute]!=REAL) {
		fprintf(stderr,"Attribute %d is not nominal at attributesInfo::getNumValuesAttribute %d\n",attribute);
		exit(1);
	}
#endif

	return deviationOfAttribute[attribute][whichClass];
}

float attributesInfo::getAverageOfAttribute(int whichClass,int attribute)
{
#ifdef DEBUG
	if(attribute<0 || attribute>=numAttributes-1) {
		fprintf(stderr,"Incorrect value at attributesInfo::getNumValuesAttribute %d\n",attribute);
		exit(1);
	}

	if(typeOfAttributes[attribute]!=REAL) {
		fprintf(stderr,"Attribute %d is not nominal at attributesInfo::getNumValuesAttribute %d\n",attribute);
		exit(1);
	}
#endif

	return averageOfAttribute[attribute][whichClass];
}

float attributesInfo::getFrequencyOfValueOfAttribute(int whichClass
	,int attribute,int value)
{
	return valueFrequenciesForNominalAttributes[attribute][whichClass][value];
}

int attributesInfo::getMostFrequentValueOfAttribute(int whichClass,int attribute)
{
#ifdef DEBUG
	if(attribute<0 || attribute>=numAttributes-1) {
		fprintf(stderr,"Incorrect value at attributesInfo::getNumValuesAttribute %d\n",attribute);
		exit(1);
	}

	if(typeOfAttributes[attribute]!=NOMINAL) {
		fprintf(stderr,"Attribute %d is not nominal at attributesInfo::getNumValuesAttribute %d\n",attribute);
		exit(1);
	}
#endif

	return mostFrequentValueForNominalAttributes[attribute][whichClass];
}

void attributesInfo::setBounds(float *min,float *max)
{
	int i;

	for(i=0;i<numAttributes-1;i++) {
		minDomain[i]=min[i];
		maxDomain[i]=max[i];
		sizeDomain[i]=max[i]-min[i];
		sizeDomain2[i]=sizeDomain[i]/2;
	}
}

int attributesInfo::valueOfNominalAttribute(int attribute,char *def) {
        int i;
        int value=-1;
        int values;
        JString tmp(def);

        if(attribute==numAttributes) values=getNumClasses();
        else values=getNumValuesAttribute(attribute);

        for(i=0;i<values && value==-1;i++) {
                if(getNominalValue(attribute,i)->equals(tmp)) value=i;
        }
	return value;
}

