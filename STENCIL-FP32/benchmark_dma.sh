#!/bin/bash

# Benchmark script for Jacobi 2D Stencil kernels
# Tests multiple dataset sizes with 5 runs each and calculates averages and speedups

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="benchmark_results_${TIMESTAMP}.txt"

# Dataset sizes to test (all datasets will be tested)
DATASETS=("MINI_DATASET" "SMALL_DATASET" "STANDARD_DATASET" "LARGE_DATASET" "EXTRALARGE_DATASET" "HUGE_DATASET" "HUMONGOUS_DATASET")

# Optional: Filter datasets by max dimension if provided as argument
MAX_DIM=${1:-16384}  # Default to 16384 (no filtering)

# Array to store results
declare -A results

echo "========================================" | tee -a "$LOG_FILE"
echo "Jacobi 2D Stencil Benchmark" | tee -a "$LOG_FILE"
echo "Date: $(date)" | tee -a "$LOG_FILE"
echo "Max dimension filter: ${MAX_DIM}" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Function to extract timing from output
extract_time() {
    local pattern=$1
    local output=$2
    # Look for "Total kernel execution time: X.XXXXXX" format (within 2 lines after pattern)
    echo "$output" | grep -A 2 "$pattern" | grep "Total kernel execution time" | awk '{print $5}'
}

# Process each dataset
for dataset in "${DATASETS[@]}"; do
    # Get dataset dimensions (TSTEPS is constant at 100 for all datasets)
    TSTEPS=100
    case $dataset in
        "MINI_DATASET")
            DIM=128
            ;;
        "SMALL_DATASET")
            DIM=256
            ;;
        "STANDARD_DATASET")
            DIM=1024
            ;;
        "LARGE_DATASET")
            DIM=2048
            ;;
        "EXTRALARGE_DATASET")
            DIM=4096
            ;;
        "HUGE_DATASET")
            DIM=8192
            ;;
        # "HUMONGOUS_DATASET")
        #     DIM=16384
        #     ;;
    esac
    
    # Skip if dimension exceeds max
    if [ $DIM -gt $MAX_DIM ]; then
        echo "Skipping $dataset (${DIM}x${DIM}) - exceeds max dimension" | tee -a "$LOG_FILE"
        continue
    fi
    
    echo "========================================" | tee -a "$LOG_FILE"
    echo "Testing $dataset (${DIM}x${DIM}, ${TSTEPS} timesteps)" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    
    # Compile with current dataset using Makefile
    echo "Compiling for $dataset..." | tee -a "$LOG_FILE"
    make -f Makefile_cudaDMA clean > /dev/null 2>&1
    make -f Makefile_cudaDMA DATASET=${dataset} 2>&1 | grep -v "warning" | tee -a "$LOG_FILE"
    
    if [ $? -ne 0 ]; then
        echo "Compilation failed for $dataset" | tee -a "$LOG_FILE"
        continue
    fi
    
    # Initialize accumulators
    sum_baseline=0
    sum_shared=0
    sum_cudadma=0
    sum_cpu=0
    
    # Run 5 times
    echo "Running 5 iterations..." | tee -a "$LOG_FILE"
    for i in {1..5}; do
        echo "  Run $i/5..." | tee -a "$LOG_FILE"
        
        # Execute and capture output
        output=$(./jacobi2D_cudaDMA 2>&1 | tee last_console.log)
        
        # Check for CPU-GPU mismatches
        mismatch_baseline=$(echo "$output" | grep "Baseline.*Non-Matching CPU-GPU Outputs" | awk '{print $NF}')
        mismatch_shared=$(echo "$output" | grep "Shared Memory.*Non-Matching CPU-GPU Outputs" | awk '{print $NF}')
        mismatch_cudadma=$(echo "$output" | grep "cudaDMA.*Non-Matching CPU-GPU Outputs" | awk '{print $NF}')
        
        # Exit with error if any mismatches found
        if [ -n "$mismatch_baseline" ] && [ "$mismatch_baseline" != "0" ]; then
            echo "ERROR: Baseline kernel has $mismatch_baseline CPU-GPU mismatches!" | tee -a "$LOG_FILE"
            echo "Benchmark aborted due to validation failure." | tee -a "$LOG_FILE"
            exit 1
        fi
        
        if [ -n "$mismatch_shared" ] && [ "$mismatch_shared" != "0" ]; then
            echo "ERROR: Shared Memory kernel has $mismatch_shared CPU-GPU mismatches!" | tee -a "$LOG_FILE"
            echo "Benchmark aborted due to validation failure." | tee -a "$LOG_FILE"
            exit 1
        fi
        
        if [ -n "$mismatch_cudadma" ] && [ "$mismatch_cudadma" != "0" ]; then
            echo "ERROR: cudaDMA kernel has $mismatch_cudadma CPU-GPU mismatches!" | tee -a "$LOG_FILE"
            echo "Benchmark aborted due to validation failure." | tee -a "$LOG_FILE"
            exit 1
        fi
        
        # Extract timings
        baseline_time=$(extract_time "GPU Time (Baseline - No Shared Memory)" "$output")
        shared_time=$(extract_time "GPU Time (Shared Memory Optimized)" "$output")
        cudadma_time=$(extract_time "GPU Time (cudaDMA Warp-Specialized)" "$output")
        cpu_time=$(extract_time "CPU Time" "$output")
        
        # Check if we got valid timings
        if [ -z "$baseline_time" ] || [ -z "$shared_time" ] || [ -z "$cudadma_time" ]; then
            echo "    Warning: Failed to extract timings from run $i" | tee -a "$LOG_FILE"
            continue
        fi
        
        # Add to sums
        sum_baseline=$(echo "$sum_baseline + $baseline_time" | bc -l)
        sum_shared=$(echo "$sum_shared + $shared_time" | bc -l)
        sum_cudadma=$(echo "$sum_cudadma + $cudadma_time" | bc -l)
        
        if [ -n "$cpu_time" ]; then
            sum_cpu=$(echo "$sum_cpu + $cpu_time" | bc -l)
        fi
        
        echo "    Baseline: ${baseline_time}s, Shared: ${shared_time}s, cudaDMA: ${cudadma_time}s, CPU: ${cpu_time}s" | tee -a "$LOG_FILE"
    done
    
    # Calculate averages
    avg_baseline=$(echo "scale=6; $sum_baseline / 5" | bc -l)
    avg_shared=$(echo "scale=6; $sum_shared / 5" | bc -l)
    avg_cudadma=$(echo "scale=6; $sum_cudadma / 5" | bc -l)
    
    # Check if we have valid CPU time
    if [ -n "$sum_cpu" ] && [ "$sum_cpu" != "0" ]; then
        avg_cpu=$(echo "scale=6; $sum_cpu / 5" | bc -l)
    else
        avg_cpu="N/A"
    fi
    
    # Calculate speedups over baseline (only if values are valid and non-zero)
    if [ -n "$avg_shared" ] && [ "$avg_shared" != "0" ] && [ "$avg_baseline" != "0" ]; then
        speedup_shared=$(echo "scale=3; $avg_baseline / $avg_shared" | bc -l)
    else
        speedup_shared="N/A"
    fi
    
    if [ -n "$avg_cudadma" ] && [ "$avg_cudadma" != "0" ] && [ "$avg_baseline" != "0" ]; then
        speedup_cudadma=$(echo "scale=3; $avg_baseline / $avg_cudadma" | bc -l)
    else
        speedup_cudadma="N/A"
    fi
    
    if [ "$avg_cpu" != "N/A" ] && [ "$avg_cpu" != "0" ] && [ "$avg_baseline" != "0" ]; then
        speedup_cpu=$(echo "scale=3; $avg_baseline / $avg_cpu" | bc -l)
    else
        speedup_cpu="N/A"
    fi
    
    # Store results
    results["${dataset}_dim"]=$DIM
    results["${dataset}_baseline"]=$avg_baseline
    results["${dataset}_shared"]=$avg_shared
    results["${dataset}_cudadma"]=$avg_cudadma
    results["${dataset}_cpu"]=$avg_cpu
    results["${dataset}_speedup_shared"]=$speedup_shared
    results["${dataset}_speedup_cudadma"]=$speedup_cudadma
    
    echo "" | tee -a "$LOG_FILE"
    echo "Average Results:" | tee -a "$LOG_FILE"
    echo "  Baseline:       ${avg_baseline}s" | tee -a "$LOG_FILE"
    echo "  Shared Memory:  ${avg_shared}s  (speedup: ${speedup_shared}x over baseline)" | tee -a "$LOG_FILE"
    echo "  cudaDMA:        ${avg_cudadma}s  (speedup: ${speedup_cudadma}x over baseline)" | tee -a "$LOG_FILE"
    echo "  CPU:            ${avg_cpu}s" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
done

# Print summary table
echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "SUMMARY TABLE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
printf "%-20s | %-10s | %-10s | %-10s | %-10s | %-10s | %-8s | %-8s\n" \
    "Dataset" "Dimensions" "Baseline(s)" "Shared(s)" "cudaDMA(s)" "CPU(s)" "Spd-Sh" "Spd-DMA" | tee -a "$LOG_FILE"
printf "%s\n" "$(printf '=%.0s' {1..120})" | tee -a "$LOG_FILE"

for dataset in "${DATASETS[@]}"; do
    dim=${results["${dataset}_dim"]}
    if [ -z "$dim" ]; then
        continue  # Skip if dataset wasn't processed
    fi
    
    baseline=${results["${dataset}_baseline"]}
    shared=${results["${dataset}_shared"]}
    cudadma=${results["${dataset}_cudadma"]}
    cpu=${results["${dataset}_cpu"]}
    speedup_shared=${results["${dataset}_speedup_shared"]}
    speedup_cudadma=${results["${dataset}_speedup_cudadma"]}
    
    printf "%-20s | %-10s | %-10s | %-10s | %-10s | %-10s | %-8s | %-8s\n" \
        "$dataset" "${dim}x${dim}" "$baseline" "$shared" "$cudadma" "$cpu" "$speedup_shared" "$speedup_cudadma" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Benchmark complete!" | tee -a "$LOG_FILE"
echo "Results saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
