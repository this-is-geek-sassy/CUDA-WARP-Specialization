# Compiler
NVCC := nvcc

# Compiler flags
# -I. allows includes like "drivers/basic_dgemm_driver.h"
# -O3 is for optimization
# -arch=sm_XX should be set to your GPU's compute capability (e.g., sm_75 for Turing)
NVCCFLAGS := -I. -O3 -arch=sm_75 --ptxas-options=-v

# Linker flags (add libraries here if needed, e.g., -lcublas)
LDFLAGS :=

# Name of the final executable
TARGET := dgemm

# --- Automatic File Discovery ---
# Find all .cpp and .cu files in the project directories
SOURCES := $(wildcard *.cpp) $(wildcard drivers/*.cu)

# Generate object file names from source file names
# e.g., drivers/dgemm.cu -> drivers/dgemm.o
OBJECTS := $(patsubst %.cpp,%.o,$(patsubst %.cu,%.o,$(SOURCES)))

# --- Build Rules ---

# The default rule: build the executable
all: $(TARGET)

# Rule to link the final executable
$(TARGET): $(OBJECTS)
	@echo "--- Linking ---"
	$(NVCC) $(OBJECTS) -o $(TARGET) $(LDFLAGS)
	@echo "Build complete: $(TARGET)"

# Rule to compile CUDA (.cu) and C++ (.cpp) source files into object files (.o)
# The '-c' flag means compile only, do not link.
# '$<' is the source file name, '$@' is the target file name.
%.o: %.cu
	@echo "Compiling CUDA: $<"
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.o: %.cpp
	@echo "Compiling C++: $<"
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# --- Utility Rules ---

# Rule to clean up build files
clean:
	@echo "--- Cleaning up ---"
	rm -f $(TARGET) $(OBJECTS)

# Phony targets are not files
.PHONY: all clean