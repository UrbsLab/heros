#include "timerSymbolicKR.h"
#include "timerMDL.h"
#include "timerGlobals.h"
#include "attributesInfo.h"
#include "messageBuffer.h"
#include <math.h>

extern messageBuffer mb;
extern attributesInfo ai;
extern timerMDL *tMDL;
extern timerGlobals *tGlobals;

timerSymbolicKR::timerSymbolicKR()
{
	if(cm.thereIsParameter(KR_ADI) || cm.thereIsParameter(KR_INSTANCE_SET) || cm.thereIsParameter(KR_HYPERRECT)) return;


	if (cm.thereIsParameter(KR_LCS)) {
		probSharp = cm.getParameter(PROB_SHARP);
	} else {
		if(cm.thereIsParameter(PROB_ONE)) {
			tGlobals->probOne = cm.getParameter(PROB_ONE);
		} else {
			int num=ai.getNumAttributesMC();
			int minR=tGlobals->minClassifiersInit;
			double nc=ai.getNumClasses();
			tGlobals->probOne=pow(1-pow(nc,-1.0/minR),1.0/num);
	                if(tGlobals->probOne<0.90) tGlobals->probOne=0.90;
	                mb.printf("Probability of ONE set to %f\n",tGlobals->probOne);
		}

		sizeAttribute = new int[ai.getNumAttributes()];
		offsetAttribute = new int[ai.getNumAttributes()];
		ruleSize = 0;
		int i;
		for (i = 0; i < ai.getNumAttributesMC(); i++) {
			if (ai.getTypeOfAttribute(i) == REAL) {
				fprintf(stderr,"This representation cannot handle real-valued attributes\n");
				exit(1);
			} else {
				sizeAttribute[i] = ai.getNumValuesAttribute(i);
			}
			offsetAttribute[i] = ruleSize;
			ruleSize += sizeAttribute[i];
		}
		sizeAttribute[i]=1;
		offsetAttribute[i]=ruleSize;
		ruleSize++;
	}
}

void timerSymbolicKR::dumpStats(int iteration)
{
}
