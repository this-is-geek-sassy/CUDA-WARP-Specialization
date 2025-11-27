# Makefile for GEMM Tensor Core Implementations
# Various optimization strategies for performance comparison

# Compiler and flags
NVCC := nvcc
ARCH := -arch=sm_89
CFLAGS := -O3

# All executables
EXECUTABLES := gemm_tensor gemm_tensor_double_buffer gemm_warp_spec_single_cpasync gemm_warp_spec_double_cpasync gemm_tf32

# Default target
all: $(EXECUTABLES)
	@echo "✓ All implementations compiled successfully!"
	@echo ""
	@echo "Available executables:"
	@echo "  - gemm_tensor                  : Baseline (FP16 with FP32 accumulation)"
	@echo "  - gemm_tensor_double_buffer    : Double buffering (no warp specialization)"
	@echo "  - gemm_warp_spec_single_cpasync: Warp specialization + single buffer + cp.async"
	@echo "  - gemm_warp_spec_double_cpasync: Warp specialization + double buffering + cp.async"
	@echo "  - gemm_tf32                    : TF32 precision (FP32 input with TF32 tensor cores)"

# Baseline - simple implementation
gemm_tensor: gemm_tensor.cu gemm_tensor.cuh
	@echo "Compiling gemm_tensor (baseline)..."
	$(NVCC) $(CFLAGS) $(ARCH) $< -o $@

# Double buffering without warp specialization
gemm_tensor_double_buffer: gemm_tensor_double_buffer.cu gemm_tensor.cuh
	@echo "Compiling gemm_tensor_double_buffer (double buffering, no warp spec)..."
	$(NVCC) $(CFLAGS) $(ARCH) $< -o $@

# Warp specialization with single buffer + cp.async
gemm_warp_spec_single_cpasync: gemm_warp_spec_single_cpasync.cu gemm_tensor.cuh
	@echo "Compiling gemm_warp_spec_single_cpasync (warp spec + single buffer + cp.async)..."
	$(NVCC) $(CFLAGS) $(ARCH) $< -o $@

# Warp specialization + double buffering + cp.async
gemm_warp_spec_double_cpasync: gemm_warp_spec_double_cpasync.cu gemm_tensor.cuh
	@echo "Compiling gemm_warp_spec_double_cpasync (warp spec + double buffer + cp.async)..."
	$(NVCC) $(CFLAGS) $(ARCH) $< -o $@

# TF32 precision (FP32 input with TF32 tensor cores)
gemm_tf32: gemm_tf32.cu gemm_tensor.cuh
	@echo "Compiling gemm_tf32 (TF32 precision)..."
	$(NVCC) $(CFLAGS) $(ARCH) $< -o $@

# Clean all executables and object files
clean:
	@echo "Cleaning..."
	rm -f $(EXECUTABLES) *.o gemm_warp_spec_single gemm_warp_spec_cpasync gemm_warp_spec_double gemm_double_buffer gemm_warp_spec_double_412 gemm_tensor_double_buffer
	@echo "✓ Clean complete"

# Test all versions
test: all
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║                    TESTING ALL VERSIONS                          ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "1. Testing baseline..."
	@./gemm_tensor 2>&1 | grep -A1 "GPU Tensor Core"
	@echo ""
	@echo "2. Testing warp spec single + cp.async..."
	@./gemm_warp_spec_single_cpasync 2>&1 | grep -A1 "GPU Tensor Core"
	@echo ""
	@echo "3. Testing warp spec + double buffer + cp.async..."
	@./gemm_warp_spec_double_cpasync 2>&1 | grep -A1 "GPU Tensor Core"
	@echo ""
	@echo "✓ All tests complete"

# Benchmark all versions (5 runs each with averages)
benchmark: all
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║              BENCHMARKING ALL VERSIONS (5 runs each)             ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "1. Benchmarking gemm_tensor (baseline)..."
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@sum=0; \
	for i in 1 2 3 4 5; do \
		time=$$(./gemm_tensor 2>&1 | grep -A1 "GPU Tensor Core" | tail -1); \
		printf "  Run %d: %ss\n" $$i "$$time"; \
		sum=$$(echo "$$sum + $$time" | bc -l); \
	done; \
	avg=$$(echo "scale=6; $$sum / 5" | bc -l); \
	ms=$$(echo "scale=3; $$avg * 1000" | bc -l); \
	printf "  Average: %ss (%.3fms)\n" "$$avg" "$$ms"
	@echo ""
	@echo "2. Benchmarking gemm_tensor_double_buffer (double buffering, no warp spec)..."
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@sum=0; \
	for i in 1 2 3 4 5; do \
		time=$$(./gemm_tensor_double_buffer 2>&1 | grep -A1 "GPU Tensor Core" | tail -1); \
		printf "  Run %d: %ss\n" $$i "$$time"; \
		sum=$$(echo "$$sum + $$time" | bc -l); \
	done; \
	avg=$$(echo "scale=6; $$sum / 5" | bc -l); \
	ms=$$(echo "scale=3; $$avg * 1000" | bc -l); \
	printf "  Average: %ss (%.3fms)\n" "$$avg" "$$ms"
	@echo ""
	@echo "3. Benchmarking gemm_warp_spec_single_cpasync (warp spec + single buffer + cp.async)..."
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@sum=0; \
	for i in 1 2 3 4 5; do \
		time=$$(./gemm_warp_spec_single_cpasync 2>&1 | grep -A1 "GPU Tensor Core" | tail -1); \
		printf "  Run %d: %ss\n" $$i "$$time"; \
		sum=$$(echo "$$sum + $$time" | bc -l); \
	done; \
	avg=$$(echo "scale=6; $$sum / 5" | bc -l); \
	ms=$$(echo "scale=3; $$avg * 1000" | bc -l); \
	printf "  Average: %ss (%.3fms)\n" "$$avg" "$$ms"
	@echo ""
	@echo "4. Benchmarking gemm_warp_spec_double_cpasync (warp spec + double buffer + cp.async)..."
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@sum=0; \
	for i in 1 2 3 4 5; do \
		time=$$(./gemm_warp_spec_double_cpasync 2>&1 | grep -A1 "GPU Tensor Core" | tail -1); \
		printf "  Run %d: %ss\n" $$i "$$time"; \
		sum=$$(echo "$$sum + $$time" | bc -l); \
	done; \
	avg=$$(echo "scale=6; $$sum / 5" | bc -l); \
	ms=$$(echo "scale=3; $$avg * 1000" | bc -l); \
	printf "  Average: %ss (%.3fms)\n" "$$avg" "$$ms"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "✓ Benchmark complete!"
	@echo ""


# Help message
help:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║            GEMM Tensor Core - Build Instructions                ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Available targets:"
	@echo "  make all       - Build all implementations"
	@echo "  make clean     - Remove all executables and object files"
	@echo "  make test      - Build and run all versions (1 run each)"
	@echo "  make benchmark - Build and run 5 iterations with averages"
	@echo "  make help      - Show this help message"
	@echo ""
	@echo "Individual targets:"
	@echo "  make gemm_tensor           - Baseline implementation"
	@echo "  make gemm_double_buffer    - Double buffering only"
	@echo "  make gemm_warp_spec_single - Warp specialization (single buffer)"
	@echo "  make gemm_warp_spec_double        - Warp spec + double buffering"
	@echo ""


.PHONY: all clean test benchmark help
