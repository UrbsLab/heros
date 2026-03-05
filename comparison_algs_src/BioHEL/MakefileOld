SUFFIX=
EXEC=biohel biohelcuda


OBJS=random.o classifier_gabil.o classifierFitness.o instanceSet.o \
	lex.yy.o instance.o timeManagement.o JString.o populationWrapper.o \
	timersManagement.o ga.o factory.o attributesInfo.o timerHierar.o \
	timerMDL.o timerSymbolicKR.o timerRealKR.o agentPerformance.o utils.o \
	timerGlobals.o timerMutation.o timerEvolutionStats.o windowingILAS.o \
	timerCrossover.o classifier_hyperrect.o mtwist.o \
	agentPerformanceTraining.o classifier_hyperrect_sse.o \
	classifier_rotated_hyperrect.o classifier_hyperrect_list.o windowingGWS.o \
	classifier_hyperrect_list_real.o classifier_hyperrect_list_discrete.o


CUDA_OBJS=functions.o kernels.cu.o random.o classifier_gabil.o classifierFitness.o instanceSet.o \
	lex.yy.o instance.o timeManagement.o JString.o populationWrapper.o \
	timersManagement.o ga.o factory.o attributesInfo.o timerHierar.o \
	timerMDL.o timerSymbolicKR.o timerRealKR.o agentPerformance.o utils.o \
	timerGlobals.o timerMutation.o timerEvolutionStats.o windowingILAS.o \
	timerCrossover.o classifier_hyperrect.o mtwist.o \
	agentPerformanceTraining.o classifier_hyperrect_sse.o \
	classifier_rotated_hyperrect.o classifier_hyperrect_list.o windowingGWS.o \
	classifier_hyperrect_list_real.o classifier_hyperrect_list_discrete.o


#CFLAGS=-O3 -m64 -march=opteron -D__CUDA_COMPILED__=1
#CFLAGS=-O3 -march=athlon-mp -D__CUDA_COMPILED__=1
CFLAGS=-O3
#-D__CUDA_COMPILED__=0
#CFLAGS=-O3 -march=pentium4 -D__CUDA_COMPILED__=1

CUDA_SDK_PATH=/usr/local/cuda/NVIDIA_GPU_Computing_SDK/CUDALibraries
CUDA_PATH=/usr/local/cuda

LDFLAGS=-fPIC  
END_LDFLAGS=-L${CUDA_PATH}/lib64 -L${CUDA_SDK_PATH}/lib -L${CUDA_SDK_PATH}/common/lib/linux -lcuda -lcudart 
CC=g++
LEX=flex
NVCC=${CUDA_PATH}/bin/nvcc
CUFLAGS=--ptxas-options=-v --compiler-options -I${CUDA_PATH}/include -I${CUDA_SDK_PATH}/common/inc -DUNIX 

default: biohel

cuda: cudabuild biohelcuda

biohel: ${OBJS} main.o
	${CC} ${LDFLAGS} ${OBJS} main.o -o biohel 

biohelcuda: ${CUDA_OBJS} main.o
	${CC} ${LDFLAGS} ${CUDA_OBJS} main.o -o biohelcuda ${END_LDFLAGS}

cudabuild:
	${NVCC} ${CUFLAGS} -o kernels.cu.o -c kernels.cu


install: ${EXEC}
	cp ${EXEC} ${HOME}/bin

dep:
	${CC} -MM ${CFLAGS} *.cpp > .depend

.cpp.o:
	${CC} ${CFLAGS} -c $<

.c.o:
	${CC} ${CFLAGS} -c $<


clean:
	rm -f *.o core ${EXEC}

lex.yy.o: lex.yy.cpp configManagement.h dictionary.h configCodes.h \
  JVector.h M95_types.h \
  attributesInfo.h JString.h instance.h random.h mt19937ar-cok.h lex_conf.l
	${LEX} -i -olex.yy.cpp lex_conf.l
	${CC} ${CFLAGS} -c lex.yy.cpp

include .depend
