# Makefile for GEMM Tensor Core Implementations
# Various optimization strategies for performance comparison

# Compiler and flags
NVCC := nvcc
ARCH := -arch=sm_89
CFLAGS := -O3

# All executables
EXECUTABLES := gemm_tesnor gemm_double_buffer gemm_warp_spec_single gemm_optimized

# Default target
all: $(EXECUTABLES)
	@echo "✓ All implementations compiled successfully!"
	@echo ""
	@echo "Available executables:"
	@echo "  - gemm_tesnor           : Baseline (simple, no optimizations)"
	@echo "  - gemm_double_buffer    : Double buffering only"
	@echo "  - gemm_warp_spec_single : Warp specialization with single buffer"
	@echo "  - gemm_optimized        : Warp specialization + double buffering"

# Baseline - simple implementation
gemm_tesnor: gemm_tesnor.cu gemm_tensor.cuh
	@echo "Compiling gemm_tesnor (baseline)..."
	$(NVCC) $(CFLAGS) $(ARCH) $< -o $@

# Double buffering only (no warp specialization)
gemm_double_buffer: gemm_double_buffer.cu gemm_tensor.cuh
	@echo "Compiling gemm_double_buffer..."
	$(NVCC) $(CFLAGS) $(ARCH) $< -o $@

# Warp specialization with single buffer
gemm_warp_spec_single: gemm_warp_spec_single.cu gemm_tensor.cuh
	@echo "Compiling gemm_warp_spec_single..."
	$(NVCC) $(CFLAGS) $(ARCH) $< -o $@

# Warp specialization + double buffering
gemm_optimized: gemm_optimized.cu gemm_tensor.cuh
	@echo "Compiling gemm_optimized..."
	$(NVCC) $(CFLAGS) $(ARCH) $< -o $@

# Clean all executables and object files
clean:
	@echo "Cleaning..."
	rm -f $(EXECUTABLES) *.o gemm_optimized_412
	@echo "✓ Clean complete"

# Test all versions
test: all
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║                    TESTING ALL VERSIONS                          ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "1. Testing baseline..."
	@./gemm_tesnor 2>&1 | grep -A1 "GPU Tensor Core"
	@echo ""
	@echo "2. Testing double buffer..."
	@./gemm_double_buffer 2>&1 | grep -A1 "GPU Tensor Core"
	@echo ""
	@echo "3. Testing warp spec single..."
	@./gemm_warp_spec_single 2>&1 | grep -A1 "GPU Tensor Core"
	@echo ""
	@echo "4. Testing optimized (warp spec + double buffer)..."
	@./gemm_optimized 2>&1 | grep -A1 "GPU Tensor Core"
	@echo ""
	@echo "✓ All tests complete"

# Benchmark all versions (5 runs each with averages)
benchmark: all
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║              BENCHMARKING ALL VERSIONS (5 runs each)             ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "1. Benchmarking gemm_tesnor (baseline)..."
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@sum=0; \
	for i in 1 2 3 4 5; do \
		time=$$(./gemm_tesnor 2>&1 | grep -A1 "GPU Tensor Core" | tail -1); \
		printf "  Run %d: %ss\n" $$i "$$time"; \
		sum=$$(echo "$$sum + $$time" | bc -l); \
	done; \
	avg=$$(echo "scale=6; $$sum / 5" | bc -l); \
	ms=$$(echo "scale=3; $$avg * 1000" | bc -l); \
	printf "  Average: %ss (%.3fms)\n" "$$avg" "$$ms"
	@echo ""
	@echo "2. Benchmarking gemm_double_buffer (double buffering only)..."
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@sum=0; \
	for i in 1 2 3 4 5; do \
		time=$$(./gemm_double_buffer 2>&1 | grep -A1 "GPU Tensor Core" | tail -1); \
		printf "  Run %d: %ss\n" $$i "$$time"; \
		sum=$$(echo "$$sum + $$time" | bc -l); \
	done; \
	avg=$$(echo "scale=6; $$sum / 5" | bc -l); \
	ms=$$(echo "scale=3; $$avg * 1000" | bc -l); \
	printf "  Average: %ss (%.3fms)\n" "$$avg" "$$ms"
	@echo ""
	@echo "3. Benchmarking gemm_warp_spec_single (warp spec, single buffer)..."
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@sum=0; \
	for i in 1 2 3 4 5; do \
		time=$$(./gemm_warp_spec_single 2>&1 | grep -A1 "GPU Tensor Core" | tail -1); \
		printf "  Run %d: %ss\n" $$i "$$time"; \
		sum=$$(echo "$$sum + $$time" | bc -l); \
	done; \
	avg=$$(echo "scale=6; $$sum / 5" | bc -l); \
	ms=$$(echo "scale=3; $$avg * 1000" | bc -l); \
	printf "  Average: %ss (%.3fms)\n" "$$avg" "$$ms"
	@echo ""
	@echo "4. Benchmarking gemm_optimized (warp spec + double buffer)..."
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@sum=0; \
	for i in 1 2 3 4 5; do \
		time=$$(./gemm_optimized 2>&1 | grep -A1 "GPU Tensor Core" | tail -1); \
		printf "  Run %d: %ss\n" $$i "$$time"; \
		sum=$$(echo "$$sum + $$time" | bc -l); \
	done; \
	avg=$$(echo "scale=6; $$sum / 5" | bc -l); \
	ms=$$(echo "scale=3; $$avg * 1000" | bc -l); \
	printf "  Average: %ss (%.3fms)\n" "$$avg" "$$ms"
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
	@echo "  make gemm_tesnor           - Baseline implementation"
	@echo "  make gemm_double_buffer    - Double buffering only"
	@echo "  make gemm_warp_spec_single - Warp specialization (single buffer)"
	@echo "  make gemm_optimized        - Warp spec + double buffering"
	@echo ""


.PHONY: all clean test benchmark help
