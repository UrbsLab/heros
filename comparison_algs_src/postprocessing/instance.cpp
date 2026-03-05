#include "instance.h"
#include "JVector.h"
#include "JString.h"
#include <math.h>
#include "attributesInfo.h"
#include "messageBuffer.h"

extern attributesInfo ai;
extern messageBuffer mb;

instance::instance(int pID,char *string,int pTraintest)
{
	int i,j;

	traintest=pTraintest;
	id=pID;
	missingValues=0;
	numAttributes=ai.getNumAttributesMC();

	//realValues=NULL;
	//if(ai.thereAreRealValuedAttributes()) {
		int num=numAttributes;
		//if(ai.onlyRealValuedAttributes()) {
			if(num%4) {
				num+=(4-num%4);
			}
		//}
		realValues=new aligned_float[num];
		bzero(realValues,num*sizeof(float));
	//}

	missing=NULL;

	//nominalValues=NULL;
	//if(ai.thereAreNominalAttributes()) {
	//	nominalValues=new unsigned char[numAttributes];
	//}

	parseInstance(string);
}

int instance::extractNominalValue(char *instance,int attribute)
{
	if(attribute<numAttributes && missingValues && missing[attribute]) return -1;
	int value=ai.valueOfNominalAttribute(attribute,instance);

	if(value==-1) {
		fprintf(stderr,"Instance:%d. Wrong nominal value of attribute %d:|%s|\n",id,attribute,instance);
		exit(1);
	}

	return value;
} 

void instance::parseInstance(char *string)
{
	int i,j;
	char *token;

	token=strtok(string,",");
	for(i=0;i<=numAttributes && token;i++) {
		if(ai.getTypeOfAttribute(i)==NOMINAL) {
			if(token[0]=='?') {
				if(missing==NULL) {
					missing = new char[numAttributes];
					for(j=0;j<numAttributes;j++) {
						missing[j]=0;
					}
				}
				missing[i]=1;
				missingValues=1;
			}
			if(i<numAttributes) {
				//nominalValues[i]=extractNominalValue(token,i);
				realValues[i]=extractNominalValue(token,i);
			} else {
				instanceClass=extractNominalValue(token,numAttributes);
			}
		} else {
			if(token[0]=='?') {
				if(missing==NULL) {
					missing = new char[numAttributes];
					for(j=0;j<numAttributes;j++) {
						missing[j]=0;
					}
				}

				missing[i]=1;
				missingValues=1;
			} else {
				float temp;
				if(sscanf(token,"%f",&temp)==0) {
					fprintf(stderr,
					  "Incorrect real value:%s\n",token);
					exit(1);
				}
				realValues[i]=temp;
			}
		}
		token=strtok(NULL,",");
	}
	if(token!=NULL) {
		fprintf(stderr,"Incorrect instance:%s\n",string);
		exit(1);
	}
}

instance::~instance()
{
	//if(nominalValues) delete nominalValues;
	if(missing) delete missing;
	if(realValues) delete realValues;
}

void instance::updateMissing()
{
	int i;

	if(!missingValues) return;

	for(i=0;i<numAttributes;i++) {
		if(missing[i]) {
			if(ai.getTypeOfAttribute(i)==REAL) {
				float val=ai.getAverageOfAttribute(instanceClass,i);
				realValues[i]=val;
				//printf("Value for attribute %d %g\n",i,val);
			} else if(ai.getTypeOfAttribute(i)==NOMINAL) {
				int val=ai.getMostFrequentValueOfAttribute(instanceClass,i);
				//nominalValues[i]=val;
				realValues[i]=val;
				//printf("Value for attribute %d %d\n",i,val);
			}
			missing[i]=0;
		}
	}

	missingValues=0;
	delete missing;
	missing=NULL;
}

void instance::dumpInstance() 
{
        int i;
        for(i=0;i<numAttributes;i++) {
                if(ai.getTypeOfAttribute(i)==NOMINAL) {
                        //mb.printf("%s,",ai.getNominalValue(i, nominalValues[i])->cstr());
                        mb.printf("%s,",ai.getNominalValue(i, (unsigned char)realValues[i])->cstr());
                } else {
                        mb.printf("%.3f,",realValues[i]);
                }
        }
        mb.printf("%s\n",ai.getNominalValue(numAttributes, instanceClass)->cstr());
}

void instance::normalize()
{
	int i;

	for(i=0;i<numAttributes;i++) {
		if(ai.getTypeOfAttribute(i)==REAL) {
			realValues[i]=(realValues[i]-ai.getMinDomain(i))/ai.getSizeDomain2(i)-1;
		}
	}
}
