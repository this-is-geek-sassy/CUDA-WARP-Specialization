# Makefile for 2D Jacobi Implementations
# Various optimization strategies for performance comparison

# Compiler and flags
NVCC := nvcc
ARCH := -arch=sm_89
CFLAGS := -O3

# All executables
EXECUTABLES := jacobi2d_baseline jacobi2d_double_buffer jacobi2d_warp_spec

# Default target
all: $(EXECUTABLES)
	@echo "✓ All implementations compiled successfully!"
	@echo ""
	@echo "Available executables:"
	@echo "  - jacobi2d_baseline      : Baseline FP32 (no optimization, 32×16 threads)"
	@echo "  - jacobi2d_double_buffer : Double buffering with 2× shared memory tiles"
	@echo "  - jacobi2d_warp_spec     : Warp specialization + double buffering"

# Baseline - 32×16 block size (512 threads per block)
jacobi2d_baseline: jacobi2D.cu jacobi2D.cuh
	@echo "Compiling jacobi2d_baseline (32×16 threads, no optimization)..."
	$(NVCC) $(CFLAGS) $(ARCH) $< -o $@

# Double buffering optimization
jacobi2d_double_buffer: jacobi2D_double_buffer.cu jacobi2D.cuh
	@echo "Compiling jacobi2d_double_buffer (double buffering)..."
	$(NVCC) $(CFLAGS) $(ARCH) $< -o $@

# Warp specialization with double buffering
jacobi2d_warp_spec: jacobi2D_warp_spec.cu jacobi2D.cuh
	@echo "Compiling jacobi2d_warp_spec (warp specialization + double buffering)..."
	$(NVCC) $(CFLAGS) $(ARCH) $< -o $@

# Clean all executables and object files
clean:
	@echo "Cleaning..."
	rm -f $(EXECUTABLES) jacobi2d_async jacobi2d_shared jacobi2d_cudadma *.o
	@echo "✓ Clean complete"

# Test all versions
test: all
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║                    TESTING ALL VERSIONS                          ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "1. Testing jacobi2d_baseline (32×16 threads)..."
	@./jacobi2d_baseline 2>&1 | grep -A1 "GPU Time"
	@echo ""
	@./jacobi2d_baseline 2>&1 | grep "Non-Matching"
	@echo ""
	@echo "2. Testing jacobi2d_double_buffer (double buffering)..."
	@./jacobi2d_double_buffer 2>&1 | grep -A1 "GPU Time"
	@echo ""
	@./jacobi2d_double_buffer 2>&1 | grep "Non-Matching"
	@echo ""
	@echo "3. Testing jacobi2d_warp_spec (warp specialization)..."
	@./jacobi2d_warp_spec 2>&1 | grep -A1 "GPU Time"
	@echo ""
	@./jacobi2d_warp_spec 2>&1 | grep "Non-Matching"
	@echo ""
	@echo "✓ All tests complete"

# Benchmark all versions (5 runs each with averages)
benchmark: all
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║              BENCHMARKING ALL VERSIONS (5 runs each)             ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "1. Benchmarking jacobi2d_baseline (32×16 threads, no optimization)..."
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@sum=0; \
	for i in 1 2 3 4 5; do \
		time=$$(./jacobi2d_baseline 2>&1 | grep -A1 "GPU Time" | tail -1); \
		printf "  Run %d: %ss\n" $$i "$$time"; \
		sum=$$(echo "$$sum + $$time" | bc -l); \
	done; \
	avg=$$(echo "scale=6; $$sum / 5" | bc -l); \
	ms=$$(echo "scale=3; $$avg * 1000" | bc -l); \
	printf "  Average: %ss (%.3fms)\n" "$$avg" "$$ms"
	@echo ""
	@echo "2. Benchmarking jacobi2d_double_buffer (double buffering)..."
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@sum=0; \
	for i in 1 2 3 4 5; do \
		time=$$(./jacobi2d_double_buffer 2>&1 | grep -A1 "GPU Time" | tail -1); \
		printf "  Run %d: %ss\n" $$i "$$time"; \
		sum=$$(echo "$$sum + $$time" | bc -l); \
	done; \
	avg=$$(echo "scale=6; $$sum / 5" | bc -l); \
	ms=$$(echo "scale=3; $$avg * 1000" | bc -l); \
	printf "  Average: %ss (%.3fms)\n" "$$avg" "$$ms"
	@echo ""
	@echo "3. Benchmarking jacobi2d_warp_spec (warp specialization + double buffering)..."
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@sum=0; \
	for i in 1 2 3 4 5; do \
		time=$$(./jacobi2d_warp_spec 2>&1 | grep -A1 "GPU Time" | tail -1); \
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
	@echo "║            2D Jacobi Stencil - Build Instructions               ║"
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
	@echo "  make jacobi2d_baseline       - Baseline implementation (32×16 threads, no optimization)"
	@echo "  make jacobi2d_double_buffer  - Double buffering with 2× shared memory tiles"
	@echo "  make jacobi2d_warp_spec      - Warp specialization (4 producer + 12 consumer warps)"
	@echo ""
	@echo "Configuration:"
	@echo "  - Grid size: 4096×4096 (STANDARD_DATASET)"
	@echo "  - Kernel: 5-point Jacobi stencil (center + 4 neighbors)"
	@echo "  - Block size: 32×16 = 512 threads per block"
	@echo "  - Iterations: 100 timesteps"
	@echo ""

.PHONY: all clean test benchmark help
