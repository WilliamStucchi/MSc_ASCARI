CXX := g++
#CXXFLAGS := -O3 -Wall -Wextra -std=c++11
CXXFLAGS := -O3 -std=c++11
LIBS := -pthread -lcasadi -lcasadi_solver

INC_DIR = inc
SRC_DIR = src
OBJ_DIR = obj

# change this to match casadi install location
CASADI_INC_DIR = /home/user/ipopt/install/include
CASADI_LIB_DIR = /home/user/ipopt/install/lib

SOURCES := $(wildcard $(SRC_DIR)/*.cpp)
INCS    := $(wildcard $(INC_DIR)/*.h)
OBJS    := $(SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

NNversion: $(OBJ_DIR)/NNversion.o $(OBJS)
			$(CXX) -o NNversion $(OBJ_DIR)/NNversion.o $(OBJS) -L$(CASADI_LIB_DIR) -L. $(LIBS) -Wl,-rpath=.

$(OBJ_DIR)/NNversion.o: main.cpp $(INCS)
			$(CXX) -c -o $@ $< -I$(INC_DIR) -I$(CASADI_INC_DIR) $(CXXFLAGS)

# -isystem treats headers as system files, and ignores warnings from them
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(INCS)
			$(CXX) -c -o $@ $< -I$(INC_DIR) -I$(CASADI_INC_DIR) $(CXXFLAGS)

.PHONY: clean
clean:
	$(RM) $(OBJS) NNversion $(OBJ_DIR)/NNversion.o
