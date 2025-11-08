# Makefile for 3D Convolution Implementations
# Various optimization strategies for performance comparison

# Compiler and flags
NVCC := nvcc
ARCH := -arch=sm_89
CFLAGS := -O3

# All executables (starting with baseline, will expand with optimizations)
EXECUTABLES := conv3d_baseline conv3d_shared conv3d_async conv3d_optimized conv3d_double_buffer conv3d_warp_spec

# Default target
all: $(EXECUTABLES)
	@echo "✓ All implementations compiled successfully!"
	@echo ""
	@echo "Available executables:"
	@echo "  - conv3d_baseline      : Baseline FP32 (no optimization, 32×32 threads)"
	@echo "  - conv3d_shared        : Shared memory optimization (3 i-slices tiled)"
	@echo "  - conv3d_async         : Async stream execution (no shared memory)"
	@echo "  - conv3d_optimized     : Shared memory + bank conflict fix (padded arrays)"
	@echo "  - conv3d_double_buffer : Double buffering only (no warp specialization)"
	@echo "  - conv3d_warp_spec     : Double buffering + warp specialization"

# Baseline - 32×32 block size (1024 threads = 32 warps per block)
conv3d_baseline: 3DConvolution.cu 3DConvolution.cuh
	@echo "Compiling conv3d_baseline (32×32 threads, no optimization)..."
	$(NVCC) $(CFLAGS) $(ARCH) $< -o $@

# Shared memory optimization - loads 3 i-slices into shared memory
conv3d_shared: 3DConvolution_shared.cu 3DConvolution.cuh
	@echo "Compiling conv3d_shared (shared memory with 3 i-slices)..."
	$(NVCC) $(CFLAGS) $(ARCH) $< -o $@

# Async stream execution
conv3d_async: 3DConvolution_async.cu 3DConvolution.cuh
	@echo "Compiling conv3d_async (async stream execution)..."
	$(NVCC) $(CFLAGS) $(ARCH) $< -o $@

# Bank conflict optimization - padded shared memory
conv3d_optimized: 3DConvolution_bank_conflicts.cu 3DConvolution.cuh
	@echo "Compiling conv3d_optimized (bank conflict fix with padding)..."
	$(NVCC) $(CFLAGS) $(ARCH) $< -o $@

# Double buffering only
conv3d_double_buffer: 3DConvolution_double_buffer.cu 3DConvolution.cuh
	@echo "Compiling conv3d_double_buffer (double buffering without warp specialization)..."
	$(NVCC) $(CFLAGS) $(ARCH) $< -o $@

# Warp specialization with double buffering
conv3d_warp_spec: 3DConvolution_warp_spec.cu 3DConvolution.cuh
	@echo "Compiling conv3d_warp_spec (double buffering + warp specialization)..."
	$(NVCC) $(CFLAGS) $(ARCH) $< -o $@

# Clean all executables and object files
clean:
	@echo "Cleaning..."
	rm -f $(EXECUTABLES) *.o
	@echo "✓ Clean complete"

# Test all versions
test: all
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║                    TESTING ALL VERSIONS                          ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "1. Testing baseline (32×32 threads)..."
	@./conv3d_baseline 2>&1 | grep -A1 "GPU Time"
	@echo ""
	@./conv3d_baseline 2>&1 | grep "Non-Matching"
	@echo ""
	@echo "2. Testing shared memory..."
	@./conv3d_shared 2>&1 | grep -A1 "GPU Time"
	@echo ""
	@./conv3d_shared 2>&1 | grep "Non-Matching"
	@echo ""
	@echo "3. Testing async streams..."
	@./conv3d_async 2>&1 | grep -A1 "GPU Time"
	@echo ""
	@./conv3d_async 2>&1 | grep "Non-Matching"
	@echo ""
	@echo "4. Testing bank conflict optimization..."
	@./conv3d_optimized 2>&1 | grep -A1 "GPU Time"
	@echo ""
	@./conv3d_optimized 2>&1 | grep "Non-Matching"
	@echo ""
	@echo "5. Testing double buffering (no warp spec)..."
	@./conv3d_double_buffer 2>&1 | grep -A1 "GPU Time"
	@echo ""
	@./conv3d_double_buffer 2>&1 | grep "Non-Matching"
	@echo ""
	@echo "6. Testing warp specialization + double buffering..."
	@./conv3d_warp_spec 2>&1 | grep -A1 "GPU Time"
	@echo ""
	@./conv3d_warp_spec 2>&1 | grep "Non-Matching"
	@echo ""
	@echo "✓ All tests complete"

# Benchmark all versions (5 runs each with averages)
benchmark: all
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║              BENCHMARKING ALL VERSIONS (5 runs each)             ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "1. Benchmarking conv3d_baseline (32×32 threads, no optimization)..."
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@sum=0; \
	for i in 1 2 3 4 5; do \
		time=$$(./conv3d_baseline 2>&1 | grep -A1 "GPU Time" | tail -1); \
		printf "  Run %d: %ss\n" $$i "$$time"; \
		sum=$$(echo "$$sum + $$time" | bc -l); \
	done; \
	avg=$$(echo "scale=6; $$sum / 5" | bc -l); \
	ms=$$(echo "scale=3; $$avg * 1000" | bc -l); \
	printf "  Average: %ss (%.3fms)\n" "$$avg" "$$ms"
	@echo ""
	@echo "2. Benchmarking conv3d_shared (shared memory with 3 i-slices)..."
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@sum=0; \
	for i in 1 2 3 4 5; do \
		time=$$(./conv3d_shared 2>&1 | grep -A1 "GPU Time" | tail -1); \
		printf "  Run %d: %ss\n" $$i "$$time"; \
		sum=$$(echo "$$sum + $$time" | bc -l); \
	done; \
	avg=$$(echo "scale=6; $$sum / 5" | bc -l); \
	ms=$$(echo "scale=3; $$avg * 1000" | bc -l); \
	printf "  Average: %ss (%.3fms)\n" "$$avg" "$$ms"
	@echo ""
	@echo "3. Benchmarking conv3d_async (async stream execution)..."
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@sum=0; \
	for i in 1 2 3 4 5; do \
		time=$$(./conv3d_async 2>&1 | grep -A1 "GPU Time" | tail -1); \
		printf "  Run %d: %ss\n" $$i "$$time"; \
		sum=$$(echo "$$sum + $$time" | bc -l); \
	done; \
	avg=$$(echo "scale=6; $$sum / 5" | bc -l); \
	ms=$$(echo "scale=3; $$avg * 1000" | bc -l); \
	printf "  Average: %ss (%.3fms)\n" "$$avg" "$$ms"
	@echo ""
	@echo "4. Benchmarking conv3d_optimized (bank conflict fix with padding)..."
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@sum=0; \
	for i in 1 2 3 4 5; do \
		time=$$(./conv3d_optimized 2>&1 | grep -A1 "GPU Time" | tail -1); \
		printf "  Run %d: %ss\n" $$i "$$time"; \
		sum=$$(echo "$$sum + $$time" | bc -l); \
	done; \
	avg=$$(echo "scale=6; $$sum / 5" | bc -l); \
	ms=$$(echo "scale=3; $$avg * 1000" | bc -l); \
	printf "  Average: %ss (%.3fms)\n" "$$avg" "$$ms"
	@echo ""
	@echo "5. Benchmarking conv3d_double_buffer (double buffering without warp spec)..."
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@sum=0; \
	for i in 1 2 3 4 5; do \
		time=$$(./conv3d_double_buffer 2>&1 | grep -A1 "GPU Time" | tail -1); \
		printf "  Run %d: %ss\n" $$i "$$time"; \
		sum=$$(echo "$$sum + $$time" | bc -l); \
	done; \
	avg=$$(echo "scale=6; $$sum / 5" | bc -l); \
	ms=$$(echo "scale=3; $$avg * 1000" | bc -l); \
	printf "  Average: %ss (%.3fms)\n" "$$avg" "$$ms"
	@echo ""
	@echo "6. Benchmarking conv3d_warp_spec (double buffering + warp specialization)..."
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@sum=0; \
	for i in 1 2 3 4 5; do \
		time=$$(./conv3d_warp_spec 2>&1 | grep -A1 "GPU Time" | tail -1); \
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
	@echo "║            3D Convolution - Build Instructions                  ║"
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
	@echo "  make conv3d_baseline       - Baseline implementation (32×32 threads, no optimization)"
	@echo "  make conv3d_shared         - Shared memory optimization (3 i-slices tiled)"
	@echo "  make conv3d_async          - Async stream execution (no shared memory)"
	@echo "  make conv3d_optimized      - Bank conflict fix (padded shared memory arrays)"
	@echo "  make conv3d_double_buffer  - Double buffering only (no warp specialization)"
	@echo "  make conv3d_warp_spec      - Double buffering + warp specialization"
	@echo ""
	@echo "Configuration:"
	@echo "  - Volume size: 768×768×768 (STANDARD_DATASET)"
	@echo "  - Kernel: 3×3×3 stencil with 9 coefficients"
	@echo "  - Block size: 32×32 = 1024 threads = 32 warps per block"
	@echo "  - Grid: 2D (j,k dimensions), loops over i dimension"
	@echo ""

.PHONY: all clean test benchmark help
