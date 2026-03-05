#include "factory.h"

#include "classifier_gabil.h"
/*#include "classifier_lcs.h"
 #include "classifier_adaptive.h"*/
#include "classifier_hyperrect.h"
#include "classifier_hyperrect_list.h"
#include "classifier_hyperrect_list_real.h"
#include "classifier_hyperrect_sse.h"
#include "classifier_rotated_hyperrect.h"
#include "classifier_hyperrect_list_discrete.h"
/*#include "classifier_instances.h"*/
#include "configManagement.h"
#include "instanceSet.h"
#include "attributesInfo.h"
#include "timerGlobals.h"

extern configManagement cm;
extern instanceSet *is;
extern attributesInfo ai;
extern timerGlobals *tGlobals;

classifierFactory::classifierFactory() {

	/*if (cm.thereIsParameter(KR_ADI))
	 classifierType = KR_ADI;*/
	/*else*/
	if (cm.thereIsParameter(KR_HYPERRECT)) {

		if (cm.thereIsParameter(HYPERRECT_LIST)) {

			if (ai.onlyRealValuedAttributes()) {

				classifierType = KR_HYPERRECT_LIST_REAL;
//			} else if (!ai.thereAreRealValuedAttributes()) {
//				classifierType = KR_HYPERRECT_LIST_DISCRETE;

			} else {
				classifierType = KR_HYPERRECT_LIST;

			}
		} else {

			if (ai.onlyRealValuedAttributes()) {

				if (cm.thereIsParameter(ROTATE_HYPERRECTANGLES)) {

					classifierType = KR_ROTATED_HYPERRECT;
				} else {

					classifierType = KR_HYPERRECT_SSE;
				}
			} else {

				classifierType = KR_HYPERRECT;
			}
		}
		/*else if (cm.thereIsParameter(KR_INSTANCE_SET))
		 classifierType = KR_INSTANCE_SET;
		 else if (cm.thereIsParameter(KR_LCS))
		 classifierType = KR_LCS;*/
	} else {

		classifierType = KR_GABIL;
	}

	//mb.printf("Classifier type: %d\n", classifierType);
}

classifier *classifierFactory::createClassifier(int numRep) {
	/*if (classifierType == KR_ADI)
	 return new classifier_adaptive();*/
	if (classifierType == KR_HYPERRECT)
		return new classifier_hyperrect();
	if (classifierType == KR_ROTATED_HYPERRECT)
		return new classifier_rotated_hyperrect();
	if (classifierType == KR_HYPERRECT_SSE)
		return new classifier_hyperrect_sse();
	if (classifierType == KR_HYPERRECT_LIST)
                return new classifier_hyperrect_list();
	if (classifierType == KR_HYPERRECT_LIST_REAL)
		return new classifier_hyperrect_list_real();
	if (classifierType == KR_HYPERRECT_LIST_DISCRETE)
                return new classifier_hyperrect_list_discrete(numRep);
	/*if (classifierType == KR_INSTANCE_SET)
	 return new classifier_instances();
	 if (classifierType == KR_LCS)
	 return new classifier_lcs();*/
	return new classifier_gabil();
}

classifier *classifierFactory::cloneClassifier(classifier * orig, int son) {
	/*if (classifierType == KR_ADI)
	 return new classifier_adaptive(
	 *((classifier_adaptive *) orig), son);*/

	if (classifierType == KR_HYPERRECT)
		return new classifier_hyperrect(*((classifier_hyperrect *) orig), son);
	if (classifierType == KR_HYPERRECT_SSE)
		return new classifier_hyperrect_sse(
				*((classifier_hyperrect_sse *) orig), son);
	if (classifierType == KR_HYPERRECT_LIST)
		return new classifier_hyperrect_list(
				*((classifier_hyperrect_list *) orig), son);
	if (classifierType == KR_HYPERRECT_LIST_REAL)
		return new classifier_hyperrect_list_real(
				*((classifier_hyperrect_list_real *) orig), son);
	if (classifierType == KR_HYPERRECT_LIST_DISCRETE)
		return new classifier_hyperrect_list_discrete(
				*((classifier_hyperrect_list_discrete *) orig), son);
	if (classifierType == KR_ROTATED_HYPERRECT)
		return new classifier_rotated_hyperrect(
				*((classifier_rotated_hyperrect *) orig), son);

	/*if (classifierType == KR_INSTANCE_SET)
	 return new classifier_instances(
	 *((classifier_instances *) orig), son);

	 if (classifierType == KR_LCS)
	 return new classifier_lcs(
	 *((classifier_lcs *) orig), son);*/

	return new classifier_gabil(*((classifier_gabil *) orig), son);
}

void classifierFactory::deleteClassifier(classifier * ind) {
	/*if (classifierType == KR_ADI)
	 delete(classifier_adaptive *) ind;
	 else*/if (classifierType == KR_HYPERRECT)
		delete (classifier_hyperrect *) ind;
	else if (classifierType == KR_HYPERRECT_SSE)
		delete (classifier_hyperrect_sse *) ind;
	else if (classifierType == KR_HYPERRECT_LIST)
		delete (classifier_hyperrect_list *) ind;
	else if (classifierType == KR_HYPERRECT_LIST_REAL)
		delete (classifier_hyperrect_list_real *) ind;
	else if (classifierType == KR_HYPERRECT_LIST_DISCRETE)
		delete (classifier_hyperrect_list_discrete *) ind;
	else if (classifierType == KR_ROTATED_HYPERRECT)
		delete (classifier_rotated_hyperrect *) ind;
	/*else if (classifierType == KR_INSTANCE_SET)
	 delete(classifier_instances *) ind;
	 else if (classifierType == KR_LCS)
	 delete(classifier_lcs *) ind;*/
	else
		delete (classifier_gabil *) ind;
}
