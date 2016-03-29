# Compiler
CC = nvcc

# (Final) Binary, Source, Include, Object and Apps directories
BIN = bin
SRC = src
INC = inc
OBJ = obj
APP = apps

# Sources for building object files, and their associated objects
SOURCES = mvstructs.cpp mmio.cpp
OBJECTS = $(patsubst %.cpp,$(OBJ)/%.o,$(SOURCES))

# Sources for the applications this makefile builds
APP_SOURCES = csrspmv.cpp 
APPS = $(patsubst %.cpp,$(BIN)/%,$(APP_SOURCES))

# Universal C flags
CFLAGS = -std=c++11 -O3 --debug --device-debug

# Get the host and shell, to determine include directories and link options
HOST = $(shell hostname)
UNAME = $(shell uname)
SKELCL_DIR = 

# Add the include directories to a single variable, which we can pass to $(CC)
INCLUDE = -I$(INC)

# Define a single general linker string for $(CC)
LINK = -lcusparse

# discover where source and header files are located
vpath %.cpp $(SRC) $(APPS)
# vpath %.cpp 
vpath %.h $(INC)
vpath %.hpp $(INC)

# General rule to trigger the others, depend on the makefile so that everything recompiles when the makefile changes
all: $(APPS) $(OBJECTS) Makefile

# Clean by simply deleting the $(OBJ) and $(BIN) directories
clean: 
	rm -rf $(OBJ) $(BIN) 

# A rule for building individual apps - make the $(BIN) directory if it doesn't exist
$(BIN)/%: $(APP)/%.cpp $(OBJECTS) 
	if [ ! -d $(BIN) ]; then mkdir $(BIN); fi
	# add -Werror here, as it breaks some of the libraries (e.g. mmio) that we rely on
	$(CC) $< $(INCLUDE) $(OBJECTS) $(CFLAGS) $(LINK) -o $@

# A rule for building individual object files - make the $(OBJ) directory if it doesn't exist
$(OBJ)/%.o: %.cpp
	if [ ! -d $(OBJ) ]; then mkdir $(OBJ); fi
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

