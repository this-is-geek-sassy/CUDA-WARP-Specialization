EXECUTABLE := gemm_fp64
CUFILES := gemm_fp64.cu

# Dataset size options:
# -DMINI_DATASET         512x512
# -DSMALL_DATASET        512x512
# -DSTANDARD_DATASET     512x512 (default)
# -DLARGE_DATASET        1024x1024
# -DEXTRALARGE_DATASET   2048x2048
DATASET := -DEXTRALARGE_DATASET

# ARCH := -gencode arch=compute_89,code=sm_89
CFLAGS := -O3

all:
	nvcc $(CFLAGS) $(DATASET) $(CUFILES) -o $(EXECUTABLE)

clean:
	rm -f *~ $(EXECUTABLE)
