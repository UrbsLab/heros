%{
#include <math.h>
#include <ctype.h>
#include <stdlib.h>

#include "configManagement.h"
#include "attributesInfo.h"
#include "random.h"
#include "messageBuffer.h"

char *charFilter(char *string);
extern configManagement cm;
extern attributesInfo ai;
extern Random rnd;
extern messageBuffer mb;


%}

DIGIT    [0-9]
LETTER   [A-Z]

%%

"NUM ATTRIBUTES "{DIGIT}+ {
/*	int atributs=atoi(charFilter(yytext));
	ai.setNumAttributes(atributs);
	mb.printf("Number of attributes in domain:%d\n"
		,atoi(charFilter(yytext)));*/
}

"PROB CROSSOVER "{DIGIT}+"."?{DIGIT}*	{
	cm.setParameter(atof(charFilter(yytext)),PROB_CROSSOVER);
	mb.printf("Crossover probability: %f\n",atof(charFilter(yytext)));
}

"POP SIZE "{DIGIT}+"."?{DIGIT}*	{
	cm.setParameter(atof(charFilter(yytext)),POP_SIZE);
	mb.printf("Popsize: %f\n",atof(charFilter(yytext)));
}

"ITERATIONS "{DIGIT}+	{
	cm.setParameter(atof(charFilter(yytext)),ITERATIONS);
	mb.printf("GA Iterations:%f\n",atof(charFilter(yytext)));
}

"INITIALIZATION MIN CLASSIFIERS "{DIGIT}+ {
	cm.setParameter(atof(charFilter(yytext))
		,INITIALIZATION_MIN_CLASSIFIERS);
	mb.printf("Minumum number of classifiers per individual in initialization:%f\n"
		,atof(charFilter(yytext)));
}
"INITIALIZATION MAX CLASSIFIERS "{DIGIT}+ {
	cm.setParameter(atof(charFilter(yytext))
		,INITIALIZATION_MAX_CLASSIFIERS);
	mb.printf("Maximum number of classifiers per individual in initialization:%f\n"
		,atof(charFilter(yytext)));
}

"IGNORE MISSING VALUES" {
	cm.setParameter(1,IGNORE_MISSING_VALUES);
	mb.printf("Ignore missing values\n");
}

"DUMP EVOLUTION STATS" {
	cm.setParameter(1,DUMP_EVOLUTION_STATS);
	mb.printf("Dump learning process statistics at each iteration\n");
}

"SELECTION ALGORITHM "{LETTER}+ {
	if(!strcasecmp(yytext+20,"TOURNAMENT")) {
		cm.setParameter(TOURNAMENT_SELECTION,SELECTION_ALGORITHM);	
		mb.printf("Tournament Selection Algorithm\n");
	} else if(!strcasecmp(yytext+20,"TOURNAMENTWOR")) {
		cm.setParameter(TOURNAMENT_WOR_SELECTION,SELECTION_ALGORITHM);	
		mb.printf("Tournament Selection without replacement Algorithm\n");
	} else if(!strcasecmp(yytext+20,"PARETO")) {
		cm.setParameter(PARETO_SELECTION,SELECTION_ALGORITHM);	
		mb.printf("Pareto Selection Algorithm\n");
	} else {
		mb.printf("Unknown selection algorithm:%s\n",yytext);
		exit(1);
	}
}

"TOURNAMENT SIZE "{DIGIT}+ {
	cm.setParameter(atof(charFilter(yytext)),TOURNAMENT_SIZE);
	mb.printf("Tournament size:%f\n",atof(charFilter(yytext)));
}									       

"SHOW FRONTS" {
	cm.setParameter(1,SHOW_FRONTS);
	mb.printf("Show Pareto Fronts\n");
}

"CROSSOVER OPERATOR "[a-zA-Z0-9]+	{
	if(!strcasecmp(&yytext[19],"1PX")) {
		cm.setParameter(CROSS_1P,CROSSOVER_OPERATOR);
		mb.printf("One Point Crossover\n");
	} else if(!strcasecmp(&yytext[19],"2PX")) {
		cm.setParameter(CROSS_2P,CROSSOVER_OPERATOR);
		mb.printf("Two Points Crossover\n");
	} else if(!strcasecmp(&yytext[19],"INFORMED")) {
		cm.setParameter(CROSS_INFORMED,CROSSOVER_OPERATOR);
		mb.printf("Informed Crossover\n");
	} else {
		mb.printf("Unknown crossover operator:%s\n",yytext);
		exit(1);
	}
}

"FITNESS FUNCTION "{LETTER}+ {
	if(!strcasecmp(yytext+17,"ACCURACY")) {
		cm.setParameter(MAXIMIZE,MAX_MIN);
		cm.setParameter(ACCURACY,FITNESS_FUNCTION);
		mb.printf("Squared accuracy fitness function\n");
	} else if(!strcasecmp(yytext+17,"MDL")) {
		cm.setParameter(MINIMIZE,MAX_MIN);
		cm.setParameter(MDL,FITNESS_FUNCTION);
		mb.printf("MDL fitness function\n");
	} else {
		mb.printf("Unknown fitness function:%s\n",yytext);
		exit(1);
	}
}

"MDL FIXED WEIGHT "{DIGIT}+"."?{DIGIT}* {
	cm.setParameter(atof(charFilter(yytext)) ,MDL_WEIGHT);
	mb.printf("MDL fixed weight %f\n" ,atof(charFilter(yytext)));
}


"MDL WEIGHT RELAX FACTOR "{DIGIT}+"."?{DIGIT}* {
	cm.setParameter(atof(charFilter(yytext)) ,MDL_WEIGHT_RELAX_FACTOR);
	mb.printf("MDL Weight relax factor %f\n" ,atof(charFilter(yytext)));
}

"MDL INITIAL TL RATIO "{DIGIT}+"."?{DIGIT}* {
	cm.setParameter(atof(charFilter(yytext)),MDL_INITIAL_TL_RATIO);
	mb.printf("Initial theory length proportion in MDL formula: %f\n" ,atof(charFilter(yytext)));
}

"MDL ITERATION "{DIGIT}+ {
	cm.setParameter(atoi(charFilter(yytext)),MDL_ITERATION);
	mb.printf("Iteracio activacio MDL %d\n",atoi(charFilter(yytext)));
}

"PRUNING ITERATION "{DIGIT}+	{
	cm.setParameter(atof(charFilter(yytext)),PRUNING_ITERATION);
	mb.printf("Pruning operator activated at iteration:%f\n",atof(charFilter(yytext)));
}

"PRUNING MIN CLASSIFIERS "{DIGIT}+ {
	cm.setParameter(atof(charFilter(yytext)),PRUNING_MIN_CLASSIFIERS);
	mb.printf("Pruning stops if #classifiers is less that %f\n",atof(charFilter(yytext)));
}

"PRUNING AUTO THRESHOLD" {
	cm.setParameter(1,PRUNING_AUTO_THRESHOLD);
	mb.printf("The number of min classifiers is automatically set\n");
}
"PRUNING AUTO THRESHOLD2" {
	cm.setParameter(1,PRUNING_AUTO_THRESHOLD2);
	mb.printf("The number of min classifiers is automatically set2\n");
}

"PRUNING AUTO OFFSET "-?{DIGIT}+ {
	cm.setParameter(atoi(charFilter(yytext)),PRUNING_AUTO_OFFSET);
	mb.printf("The min classifiers offset %d\n",atoi(charFilter(yytext)));
}

"PROB INDIVIDUAL MUTATION "{DIGIT}+"."?{DIGIT}* {
	cm.setParameter(atof(charFilter(yytext)),PROB_INDIVIDUAL_MUTATION);
	mb.printf("Individual-wise mutation probability:%f\n",atof(charFilter(yytext)));
}

"HIERARCHICAL SELECTION ITERATION "{DIGIT}+ {
	cm.setParameter(atof(charFilter(yytext)) ,HIERARCHICAL_SELECTION_ITERATION);
	mb.printf("Hierarchical selection activated, starting at iteration %d\n"
		,atoi(charFilter(yytext)));
}

"HIERARCHICAL SELECTION THRESHOLD "{DIGIT}+"."?{DIGIT}* {
	cm.setParameter(atof(charFilter(yytext)),HIERARCHICAL_SELECTION_THRESHOLD);
	mb.printf("Hierarchical selection threshold :%f\n"
		,atof(charFilter(yytext)));
}

"HIERARCHICAL SELECTION USES MDL THEORY LENGTH" {
	cm.setParameter(1,HIERARCHICAL_SELECTION_USES_MDL);
	mb.printf("Hierarchical selection uses MDL Theory Length\n");
}

"CHECK WINDOWING" {
	cm.setParameter(1,CHECK_WINDOWING);
	mb.printf("Performance tests of windowing enabled");
}

"WINDOWING ILAS "{DIGIT}+ {
	cm.setParameter(atoi(charFilter(yytext)),WINDOWING_ILAS);
	mb.printf("ILAS Windowing of degree %d\n"
		,atoi(charFilter(yytext)));
}
"WINDOWING GWS "{DIGIT}+ {
	cm.setParameter(atoi(charFilter(yytext)),WINDOWING_GWS);
	mb.printf("GWS Windowing of degree %d\n"
		,atoi(charFilter(yytext)));
}

"PROB ONE "{DIGIT}+"."?{DIGIT}* {
	cm.setParameter(atof(charFilter(yytext)),PROB_ONE);
	mb.printf("Probability of value ONE for GABIL and ADI KR:%f\n"
		,atof(charFilter(yytext)));
}

"PROB SHARP "{DIGIT}+"."?{DIGIT}* {
	cm.setParameter(atof(charFilter(yytext)),PROB_SHARP);
	mb.printf("Probability of value Sharp for LCS/Instances KR:%f\n"
		,atof(charFilter(yytext)));
}

"KR ADI" {
	cm.setParameter(1,KR_ADI);
	mb.printf("Using Adaptive Discretization Intervals Knowledge Representation\n");
}

"PROB MERGE "{DIGIT}+"."?{DIGIT}* {
	cm.setParameter(atof(charFilter(yytext)) ,PROB_MERGE);
	mb.printf("Probability of merge operator in ADI KR: %f\n"
		,atof(charFilter(yytext)));
}

"PROB SPLIT "{DIGIT}+"."?{DIGIT}* {
	cm.setParameter(atof(charFilter(yytext)) ,PROB_SPLIT);
	mb.printf("Probability of split operator in ADI KR: %f\n"
		,atof(charFilter(yytext)));
}

"PROB REINITIALIZE "{DIGIT}+"."?{DIGIT}* {
	cm.setParameter(atof(charFilter(yytext)) ,PROB_REINITIALIZE);
	mb.printf("Probability of reinitialize operator in ADI KR: %f\n"
		,atof(charFilter(yytext)));
}

"PROB REINITIALIZE AT END "{DIGIT}+"."?{DIGIT}* {
	cm.setParameter(atof(charFilter(yytext)) ,PROB_REINITIALIZE_AT_END);
	mb.printf("Probability of reinitialize operator at final iteration in ADI KR:%f\n"
		,atof(charFilter(yytext)));
}

"MAX INTERVALS "{DIGIT}+ {
	cm.setParameter(atoi(charFilter(yytext)),MAX_INTERVALS);
	mb.printf("Maximum number of intervals per attribute in ADI KR:%d\n"
		,atoi(charFilter(yytext)));
}

"KR HYPERRECT" {
	cm.setParameter(1,KR_HYPERRECT);
	mb.printf("Using HYPERRECT Knowledge Representation\n");
}

"KR LCS" {
	cm.setParameter(1,KR_LCS);
	mb.printf("Using LCS Knowledge Representation\n");
}

"KR INSTANCE SET" {
	cm.setParameter(1,KR_INSTANCE_SET);
	mb.printf("Using Instance Set/1-NN Knowledge Representation\n");
}

"ALPHA OF BLX "{DIGIT}+"."?{DIGIT}* {
	cm.setParameter(atof(charFilter(yytext)) ,ALPHA_OF_BLX);
	mb.printf("Using BLX crossover with alpha:%f\n" 
		,atof(charFilter(yytext)));
}

"D OF FR "{DIGIT}+"."?{DIGIT}* {
	cm.setParameter(atof(charFilter(yytext)) ,D_OF_FR);
	mb.printf("Using FR crossover with D:%f\n" ,atof(charFilter(yytext)));
}

"N OF SBX "{DIGIT}+"."?{DIGIT}* {
	cm.setParameter(atof(charFilter(yytext)) ,N_OF_SBX);
	mb.printf("Using SBX crossover with N:%f\n" ,atof(charFilter(yytext)));
}

"RANDOM SEED "{DIGIT}+ {
	rnd.setSeed((unsigned long int)atof(charFilter(yytext)));
	mb.printf("Random seed specified:%s\n",yytext+12);
}

"PENALIZE INDIVIDUALS WITH LESS CLASSIFIERS THAN "{DIGIT}+ {
	cm.setParameter(atoi(charFilter(yytext)),PENALIZE_MIN_SIZE);
	mb.printf("Penalize the individuals that have a size less than %d\n"
		,atoi(charFilter(yytext)));
}

"PENALIZE INDIVIDUALS WITH LESS CLASSIFIERS AT END THAN "{DIGIT}+ {
	cm.setParameter(atoi(charFilter(yytext)),PENALIZE_MIN_SIZE_AT_END);
	mb.printf("Penalize the individuals that have a size less than %d at end\n"
		,atoi(charFilter(yytext)));
}

"PARETO SELECTION ITERATION "{DIGIT}+ {
	cm.setParameter(atof(charFilter(yytext)) ,PARETO_SELECTION_ITERATION);
	mb.printf("Pareto selection activated, starting at iteration %d\n"
		,atoi(charFilter(yytext)));
}


"TOTAL TIME "{DIGIT}+ {
	cm.setParameter(atoi(charFilter(yytext)) ,TOTAL_TIME);
	mb.printf("Time spent on the learning process %d\n"
		,atoi(charFilter(yytext)));
}

"DEFAULT CLASS "[A-Z0-9]+ {
	if(!strcasecmp(yytext+14,"MAJOR")) {
		cm.setParameter(MAJOR,DEFAULT_CLASS);
		mb.printf("Majoritarian class will be default\n");
	} else if(!strcasecmp(yytext+14,"MINOR")) {
		cm.setParameter(MINOR,DEFAULT_CLASS);
		mb.printf("Minoritarian class will be default\n");
	} else if(!strcasecmp(yytext+14,"DISABLED")) {
		cm.setParameter(DISABLED,DEFAULT_CLASS);
		mb.printf("Default class disabled\n");
	} else if(!strcasecmp(yytext+14,"AUTO")) {
		cm.setParameter(AUTO,DEFAULT_CLASS);
		mb.printf("Automatical determination of default class\n");
	} else if(!strcasecmp(yytext+14,"AUTO2")) {
		cm.setParameter(AUTO,DEFAULT_CLASS);
		cm.setParameter(1,HARD_NICHING_DISABLE);
		mb.printf("Automatical determination of default class with alternative niching disabling code\n");
         } else if(!strcasecmp(yytext+14,"FIXED")) {
                cm.setParameter(FIXED,DEFAULT_CLASS);
                mb.printf("Default class fixed\n");
         } else {
                mb.printf("Unknown default class policy:%s\n",yytext);
                exit(1);
	}
}

"FIXED DEFAULT CLASS "{DIGIT}+ {
        cm.setParameter(atoi(charFilter(yytext)) ,FIXED_DEFAULT_CLASS);
        mb.printf("User fixed default class %d\n"
                   ,atoi(charFilter(yytext)));
}

"SMART INIT" {
	cm.setParameter(1 ,SMART_INIT);
	mb.printf("Initialization uses examples to create the initial rules\n");
}

"CLASS WISE INIT" {
	cm.setParameter(1 ,CLASS_WISE_INIT);
	mb.printf("Instances used in initialization are sampled with uniform class distribution\n");
}

"CLASS WISE ACC" {
	cm.setParameter(1 ,CLASS_WISE_ACC);
	mb.printf("Training accuracy computation will be class-wise\n");
}


"DUMP ACTIVATION" {
	cm.setParameter(1 ,DUMP_ACTIVATION);
	mb.printf("Dump average activation after initialization\n");
}


"PRUNING METHOD "{LETTER}+ {
	 if(!strcasecmp(yytext+15,"FTB")) {
		cm.setParameter(FTB,PRUNING_POLICY);
		mb.printf("Rule pruning policy is front to back\n");
	} else if(!strcasecmp(yytext+15,"BTF")) {
		cm.setParameter(BTF,PRUNING_POLICY);
		mb.printf("Rule pruning policy is back to front\n");
	} else if(!strcasecmp(yytext+15,"RANDOM")) {
		cm.setParameter(RANDOM,PRUNING_POLICY);
		mb.printf("Rule pruning policy is random\n");
	} else {
		mb.printf("Unknown pruning policy:%s\n",yytext);
		exit(1);
	}
}


"PROB SMART CROSSOVER "{DIGIT}+"."?{DIGIT}*    {
	cm.setParameter(atof(charFilter(yytext)),PROB_SMART_CROSSOVER);
	mb.printf("Smart crossover probability: %f\n",atof(charFilter(yytext)));
}

"NUM PARENTS SMART CROSSOVER "{DIGIT}+ {
	cm.setParameter(atof(charFilter(yytext)),NUM_PARENTS_SMART_CROSSOVER);
	mb.printf("Number of parents in smart crossover: %d\n",atoi(charFilter(yytext)));
}


"FILTER SMART CROSSOVER "{DIGIT}+"."?{DIGIT}*    {
	cm.setParameter(atof(charFilter(yytext)),FILTER_SMART_CROSSOVER);
	mb.printf("Smart crossover filter threshold: %f\n",atof(charFilter(yytext)));
}


"ADD RULES SMART CROSSOVER" {
	cm.setParameter(1,ADD_RULES_SMART_CROSSOVER);
	mb.printf("Smart crossover adds new rules\n");
}

"REPETITIONS OF RULE ORDERING "{DIGIT}+ {
	cm.setParameter(atof(charFilter(yytext)),REPETITIONS_RULE_ORDERING);
	mb.printf("Number of repetitions of the rule ordering process in SmartX: %d\n",atoi(charFilter(yytext)));
}

"ELITISM WITH SMART CROSSOVER" {
	cm.setParameter(1,ELITISM_WITH_SMART_CROSSOVER);
	mb.printf("Elitism stage will use smart crossover\n");
}
"ELITISM LAST ITERATION WITH SMART CROSSOVER" {
	cm.setParameter(1,ELITISM_LAST_ITERATION_WITH_SMART_CROSSOVER);
	mb.printf("Last iteration of elitism stage will use smart crossover\n");
}


"RULE CLEANING PROB "{DIGIT}+"."?{DIGIT}*    {
	cm.setParameter(atof(charFilter(yytext)),RULE_CLEANING_PROB);
	mb.printf("Rule cleaning probability : %f\n",atof(charFilter(yytext)));
}

"RULE GENERALIZING PROB "{DIGIT}+"."?{DIGIT}*    {
	cm.setParameter(atof(charFilter(yytext)),RULE_GENERALIZING_PROB);
	mb.printf("Rule generalizing probability : %f\n",atof(charFilter(yytext)));
}


"DUMP GENOTYPE OF BEST INDIVIDUAL EVERY ITERATIONS "{DIGIT}+ {
	cm.setParameter(atof(charFilter(yytext)),DUMP_GENOTYPE_ITERATIONS);
	mb.printf("Genotype of best individual is dumped every %d iterations\n",atoi(charFilter(yytext)));
}

"BUFFERED OUTPUT" {
	mb.enable();
}

"INFORMED CROSSOVER" {
	mb.printf("Crossover will use informed cut points (cutPoints.dat)\n");
	cm.setParameter(1,INFORMED_CROSSOVER);
}

"COVERAGE INIT "{DIGIT}+"."?{DIGIT}*    {
	cm.setParameter(atof(charFilter(yytext)),COVERAGE_INIT);
	mb.printf("Coverage ratio in initialization : %f\n",atof(charFilter(yytext)));
}

"NUM EXPRESSED ATTRIBUTES INIT "{DIGIT}+    {
	cm.setParameter(atof(charFilter(yytext)),EXPRESSED_ATT_INIT);
	mb.printf("Number of expressed attributes in initialization : %f\n",atof(charFilter(yytext)));
}


"COVERAGE BREAKPOINT "{DIGIT}+"."?{DIGIT}*    {
	cm.setParameter(atof(charFilter(yytext)),COVERAGE_BREAKPOINT);
	mb.printf("Coverage breakpoint for MDL fitness : %f\n",atof(charFilter(yytext)));
}

"COVERAGE RATIO "{DIGIT}+"."?{DIGIT}*    {
	cm.setParameter(atof(charFilter(yytext)),COVERAGE_RATIO);
	mb.printf("Coverage ratio for MDL fitness : %f\n",atof(charFilter(yytext)));
}


"REPETITIONS OF RULE LEARNING "{DIGIT}+ {
	cm.setParameter(atof(charFilter(yytext)),REPETITIONS_RULE_LEARNING);
	mb.printf("Number of times we will try to learn a rule from the current training set: %d\n",atoi(charFilter(yytext)));
}

"ROTATE HYPERRECTANGLES" {
	mb.printf("Intervalar representation will use rotations\n");
	cm.setParameter(1,ROTATE_HYPERRECTANGLES);
}

"RESTRICTED ROTATED ATTRIBUTES" {
	mb.printf("Only a subset of attributes will be rotated (rotatedAttributes.dat)\n");
	cm.setParameter(1,RESTRICTED_ROTATED_ATTRIBUTES);
}


"PROB ZERO ANGLE INIT "{DIGIT}+"."?{DIGIT}*    {
	cm.setParameter(atof(charFilter(yytext)),PROB_0ANGLE_INIT);
	mb.printf("Probability of setting an angle to 0 degrees in initialization: %f\n",atof(charFilter(yytext)));
}

"PROB ZERO ANGLE MUT "{DIGIT}+"."?{DIGIT}*    {
	cm.setParameter(atof(charFilter(yytext)),PROB_0ANGLE_MUT);
	mb.printf("Probability of setting an angle to 0 degrees in mutation: %f\n",atof(charFilter(yytext)));
}

"HYPERRECTANGLE USES LIST OF ATTRIBUTES" {
	mb.printf("Hyperrectangle attribute list knowledge representation\n");
	cm.setParameter(1,HYPERRECT_LIST);
}

"PROB GENERALIZE LIST "{DIGIT}+"."?{DIGIT}*    {
	cm.setParameter(atof(charFilter(yytext)),PROB_GENERALIZE_LIST);
	mb.printf("Probability of generalizing the hyperrect list KR: %f\n",atof(charFilter(yytext)));
}
"PROB SPECIALIZE LIST "{DIGIT}+"."?{DIGIT}*    {
	cm.setParameter(atof(charFilter(yytext)),PROB_SPECIALIZE_LIST);
	mb.printf("Probability of specializing the hyperrect list KR: %f\n",atof(charFilter(yytext)));
}

"USE COVERAGE BREAK HEURISTIC" {
	mb.printf("Using the coverage breakpoint adjustment heuristic\n");
        cm.setParameter(1,COVERAGE_BREAK_HEURISTIC);
}


"ITERATIONS OF COV BREAK ADJUSTEMENT "{DIGIT}+ {
	cm.setParameter(atof(charFilter(yytext)),ITERATIONS_COVADJ);
	mb.printf("Number of iterations of the coverage break adjustement heuristic: %d\n",atoi(charFilter(yytext)));
}

"POPULATIONS OF COV BREAK ADJUSTEMENT "{DIGIT}+ {
	cm.setParameter(atof(charFilter(yytext)),POPULATIONS_COVADJ);
	mb.printf("Number of populations used in the coverage break adjustement heuristic: %d\n",atoi(charFilter(yytext)));
}

"EXPECTED NUMBER OF ATTRIBUTES "{DIGIT}+ {
        cm.setParameter(atof(charFilter(yytext)),NUM_ATTS_K);
        mb.printf("Expected number of attributes: %d\n",atoi(charFilter(yytext)));
}

"DEVICE SELECTED "{DIGIT}+ {
        cm.setParameter(atof(charFilter(yytext)),DEVICE_SELECTED);
        mb.printf("Device selected through configuration: %f\n",atof(charFilter(yytext)));

}

"DEVICE MEMORY USED "{DIGIT}+"."?{DIGIT}* {
        cm.setParameter(atof(charFilter(yytext)),PERC_DEVICE_MEM);
        mb.printf("Percentage of device memory used: %f\n",atof(charFilter(yytext)));

}

"CUDA ENABLED" {
        mb.printf("CUDA Enabled fitness function activated\n");
        cm.setParameter(1,CUDA_ENABLED);
}





	
#.*$     /* eat up one-line comments */

[ \t\n]+	  /* eat up whitespace */

.	   mb.printf( "Unrecognized character: %s\n", yytext );

%%
int yywrap ()
{
	return 1;
}

char *charFilter(char *string)
{
	while(*string && !(isdigit(*string) || *string=='-')) string++;
	return string;
}

void parseConfig(char *configFile)
{
	int i;

	yyin = fopen( configFile, "r" );
	yylex();
}								 
