# Compiler
NVCC := nvcc

# Build Directory
BUILD_DIR := bin

# Compiler flags
# -I. allows includes like "drivers/basic_dgemm_driver.h"
# -O3 is for optimization
# -arch=sm_XX should be set to your GPU's compute capability (e.g., sm_75 for Turing)
NVCCFLAGS := -I. -O3 --ptxas-options=-v

# Linker flags (add libraries here if needed, e.g., -lcublas)
LDFLAGS :=

# Name of the final executable, placed in the build directory
TARGET := $(BUILD_DIR)/dgemm

# --- Automatic File Discovery ---
# Find all .cpp and .cu files recursively in the current directory and subdirectories
SOURCES := $(shell find . -name '*.cpp' -o -name '*.cu')

# Generate object file names and map them to the build directory
# e.g., drivers/dgemm.cu -> bin/drivers/dgemm.o
# ./main.cpp -> bin/./main.o
OBJECTS := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(patsubst %.cu,$(BUILD_DIR)/%.o,$(SOURCES)))

# --- Build Rules ---

# The default rule: build the executable
all: $(TARGET)

# Rule to link the final executable
$(TARGET): $(OBJECTS)
	@echo "--- Linking ---"
	$(NVCC) $(OBJECTS) -o $(TARGET) $(LDFLAGS)
	@echo "Build complete: $(TARGET)"

# Rule to compile CUDA (.cu) source files into object files (.o) in the BUILD_DIR
# It automatically creates subdirectories within bin as needed (e.g., bin/drivers)
$(BUILD_DIR)/%.o: %.cu
	@echo "Compiling CUDA: $< -> $@"
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Rule to compile C++ (.cpp) source files into object files (.o) in the BUILD_DIR
$(BUILD_DIR)/%.o: %.cpp
	@echo "Compiling C++: $< -> $@"
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# --- Utility Rules ---

# Rule to clean up build files by removing the entire build directory
clean:
	@echo "--- Cleaning up ---"
	rm -rf $(BUILD_DIR)

# Phony targets are not files
.PHONY: all clean
