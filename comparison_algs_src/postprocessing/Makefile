# -----------------------------
# Postprocess (CPU-only) Makefile
# -----------------------------

EXEC := postprocess

CXX  := g++
LEX  := flex

# Legacy-friendly settings (closest to old GCC 4.x era behavior)
CXXFLAGS := -O3 -fPIC -std=gnu++98

# If your legacy code needs extra permissiveness, uncomment:
# CXXFLAGS += -fpermissive
# CXXFLAGS += -Wno-narrowing -Wno-deprecated -Wno-write-strings

LDFLAGS :=
LDLIBS  :=

# -----------------------------
# Objects
# -----------------------------
OBJS := random.o classifier_gabil.o classifier_hyperrect_list.o classifierFitness.o \
	postprocessingOper.o instanceSet.o \
	lex.yy.o instance.o timeManagement.o JString.o populationWrapper.o \
	timersManagement.o ga.o factory.o attributesInfo.o timerHierar.o \
	timerMDL.o timerSymbolicKR.o timerRealKR.o agentPerformance.o utils.o \
	timerGlobals.o timerMutation.o timerEvolutionStats.o windowingILAS.o \
	timerCrossover.o classifier_hyperrect.o mtwist.o \
	agentPerformanceTraining.o classifier_hyperrect_sse.o \
	classifier_rotated_hyperrect.o windowingGWS.o \
	classifier_hyperrect_list_real.o classifier_hyperrect_list_discrete.o \
	main.o

# -----------------------------
# Default target
# -----------------------------
.PHONY: all
all: $(EXEC)

# -----------------------------
# Link
# -----------------------------
$(EXEC): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $(OBJS) $(LDLIBS)

# -----------------------------
# Compile rules
# -----------------------------
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.c
	$(CXX) $(CXXFLAGS) -c $< -o $@

# -----------------------------
# Flex rule (regenerates lex.yy.cpp and compiles it)
# -----------------------------
lex.yy.o: lex.yy.cpp configManagement.h dictionary.h configCodes.h \
  JVector.h M95_types.h attributesInfo.h JString.h instance.h random.h \
  mt19937ar-cok.h lex_conf.l
	$(LEX) -i -olex.yy.cpp lex_conf.l
	$(CXX) $(CXXFLAGS) -c lex.yy.cpp -o lex.yy.o

lex.yy.cpp: lex_conf.l
	$(LEX) -i -olex.yy.cpp lex_conf.l

# -----------------------------
# Dependency generation (optional)
# -----------------------------
.PHONY: dep
dep:
	$(CXX) -MM $(CXXFLAGS) *.cpp > .depend

-include .depend

# -----------------------------
# Install
# -----------------------------
.PHONY: install
install: $(EXEC)
	cp $(EXEC) $(HOME)/bin

# -----------------------------
# Clean
# -----------------------------
.PHONY: clean
clean:
	rm -f *.o core $(EXEC) lex.yy.cpp .depend
