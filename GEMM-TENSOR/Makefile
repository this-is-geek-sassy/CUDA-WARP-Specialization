EXECUTABLE := gemm_tensor
CUFILES := gemm_tensor.cu

all:
	nvcc -O3 $(CUFILES) -o $(EXECUTABLE) -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86

clean:
	rm -f *~ $(EXECUTABLE)
