#ifndef _SELECTION_
#define _SELECTION_

void selectionAlgorithm();

void TournamentSelectionWOR();
int selectNicheWOR(int *quotas,int num);
int selectCandidateWOR(JVector<int> &pool,int whichNiche);
void initPool(JVector<int> &pool,int whichNiche);

void TournamentSelection();
int selectNiche(int *quotas,int num);
int selectCandidate(int niche);

double ParetoOrder(double *a, double *b);
void ParetoSwap(double *a, double *b);
void ParetoQSort(double **objectives, int left, int right);
double ParetoDistance(double *a, double *b);
double ParetoSharing(double val);
void ParetoElitistTournament(double *paretoAvals);
double *ParetoFitness(double **objectives,JVector<int>&front,JVector<int>&elitismGap,int step);
void ParetoSelection(void);
void ScaleObjectives(double **objectives,int num);

JVector<classifier *> oldBests;
JVector<classifier *> oldFront;
int selectionAlg;
int tournamentSize;
int showFronts;


#endif
