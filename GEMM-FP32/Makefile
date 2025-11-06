EXECUTABLE := gemm_fp_32_baseline
CUFILES := gemm_fp32_baseline.cu

# Dataset size options:
# -DMINI_DATASET         512x512
# -DSMALL_DATASET        512x512
# -DSTANDARD_DATASET     512x512 (default)
# -DLARGE_DATASET        1024x1024
# -DEXTRALARGE_DATASET   2048x2048
# -DHUGE_DATASET         4096x4096
DATASET := -DHUGE_DATASET

ARCH := -gencode arch=compute_86,code=sm_86
CFLAGS := -O3

all:
	nvcc $(CFLAGS) $(DATASET) $(CUFILES) -o $(EXECUTABLE) $(ARCH)

clean:
	rm -f *~ $(EXECUTABLE)
