#!/bin/bash

# Benchmark script for Jacobi 2D Baseline Stencil kernel
# Tests multiple dataset sizes with 5 runs each and calculates averages

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="benchmark_baseline_results_${TIMESTAMP}.txt"

# Dataset sizes to test
DATASETS=("MINI_DATASET" "SMALL_DATASET" "STANDARD_DATASET" "LARGE_DATASET" "EXTRALARGE_DATASET")

# Optional: Filter datasets by max dimension if provided as argument
MAX_DIM=${1:-16384}  # Default to 16384 (no filtering)

# Array to store results
declare -A results

echo "========================================" | tee -a "$LOG_FILE"
echo "Jacobi 2D Baseline Stencil Benchmark" | tee -a "$LOG_FILE"
echo "Date: $(date)" | tee -a "$LOG_FILE"
echo "Max dimension filter: ${MAX_DIM}" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Function to extract timing from output
extract_time() {
    local pattern=$1
    local output=$2
    # Look for "Total kernel execution time: X.XXXXXX" format
    echo "$output" | grep -A 1 "$pattern" | grep "Total kernel execution time" | awk '{print $5}'
}

# Process each dataset
for dataset in "${DATASETS[@]}"; do
    # Get dataset dimensions
    case $dataset in
        "MINI_DATASET")
            DIM=128
            TSTEPS=20
            ;;
        "SMALL_DATASET")
            DIM=256
            TSTEPS=40
            ;;
        "STANDARD_DATASET")
            DIM=1024
            TSTEPS=100
            ;;
        "LARGE_DATASET")
            DIM=2048
            TSTEPS=200
            ;;
        "EXTRALARGE_DATASET")
            DIM=4096
            TSTEPS=500
            ;;
        "HUGE_DATASET")
            DIM=8192
            TSTEPS=1000
            ;;
        "HUMONGOUS_DATASET")
            DIM=16384
            TSTEPS=2000
            ;;
    esac
    
    # Skip if dimension exceeds max
    if [ $DIM -gt $MAX_DIM ]; then
        echo "Skipping $dataset (${DIM}x${DIM}) - exceeds max dimension" | tee -a "$LOG_FILE"
        continue
    fi
    
    echo "========================================" | tee -a "$LOG_FILE"
    echo "Testing $dataset (${DIM}x${DIM}, ${TSTEPS} timesteps)" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    
    # Compile with current dataset
    echo "Compiling for $dataset..." | tee -a "$LOG_FILE"
    nvcc -O3 -arch=sm_86 -D${dataset} -I../.. jacobi2D_baseline.cu -o jacobi2D_baseline 2>&1 | grep -v "warning" | tee -a "$LOG_FILE"
    
    if [ $? -ne 0 ]; then
        echo "Compilation failed for $dataset" | tee -a "$LOG_FILE"
        continue
    fi
    
    # Initialize accumulators
    sum_baseline=0
    sum_shared=0
    sum_texture=0
    sum_hybrid=0
    sum_cpu=0
    
    # Run 5 times
    echo "Running 5 iterations..." | tee -a "$LOG_FILE"
    for i in {1..5}; do
        echo "  Run $i/5..." | tee -a "$LOG_FILE"
        
        # Execute and capture output
        output=$(./jacobi2D_baseline 2>&1 | tee baseline_console.log)
        
        # Check for CPU-GPU mismatches (check all validation lines)
        mismatches=$(echo "$output" | grep "Non-Matching CPU-GPU Outputs" | awk '{print $NF}')
        
        # Check if any mismatch is non-zero
        has_error=0
        while IFS= read -r mismatch; do
            if [ -n "$mismatch" ] && [ "$mismatch" != "0" ]; then
                has_error=1
                echo "ERROR: Found $mismatch CPU-GPU mismatches!" | tee -a "$LOG_FILE"
            fi
        done <<< "$mismatches"
        
        if [ $has_error -eq 1 ]; then
            echo "Benchmark aborted due to validation failure." | tee -a "$LOG_FILE"
            exit 1
        fi
        
        # Extract timings
        baseline_time=$(extract_time "GPU Time (Baseline - No Shared Memory)" "$output")
        shared_time=$(extract_time "GPU Time (Shared Memory Optimized)" "$output")
        texture_time=$(extract_time "GPU Time (Texture Memory)" "$output")
        hybrid_time=$(extract_time "GPU Time (Texture + Shared Memory Hybrid)" "$output")
        cpu_time=$(extract_time "CPU Time" "$output")
        
        # Check if we got valid timings
        if [ -z "$baseline_time" ] || [ -z "$shared_time" ] || [ -z "$texture_time" ] || [ -z "$hybrid_time" ]; then
            echo "    Warning: Failed to extract timings from run $i" | tee -a "$LOG_FILE"
            continue
        fi
        
        # Add to sums
        sum_baseline=$(echo "$sum_baseline + $baseline_time" | bc -l)
        sum_shared=$(echo "$sum_shared + $shared_time" | bc -l)
        sum_texture=$(echo "$sum_texture + $texture_time" | bc -l)
        sum_hybrid=$(echo "$sum_hybrid + $hybrid_time" | bc -l)
        
        if [ -n "$cpu_time" ]; then
            sum_cpu=$(echo "$sum_cpu + $cpu_time" | bc -l)
        fi
        
        echo "    Baseline: ${baseline_time}s, Shared: ${shared_time}s, Texture: ${texture_time}s, Hybrid: ${hybrid_time}s, CPU: ${cpu_time}s" | tee -a "$LOG_FILE"
    done
    
    # Calculate averages
    avg_baseline=$(echo "scale=6; $sum_baseline / 5" | bc -l)
    avg_shared=$(echo "scale=6; $sum_shared / 5" | bc -l)
    avg_texture=$(echo "scale=6; $sum_texture / 5" | bc -l)
    avg_hybrid=$(echo "scale=6; $sum_hybrid / 5" | bc -l)
    
    # Check if we have valid CPU time
    if [ -n "$sum_cpu" ] && [ "$sum_cpu" != "0" ]; then
        avg_cpu=$(echo "scale=6; $sum_cpu / 5" | bc -l)
    else
        avg_cpu="N/A"
    fi
    
    # Calculate speedups (baseline as reference)
    if [ -n "$avg_shared" ] && [ "$avg_shared" != "0" ] && [ "$avg_baseline" != "0" ]; then
        speedup_shared=$(echo "scale=3; $avg_baseline / $avg_shared" | bc -l)
    else
        speedup_shared="N/A"
    fi
    
    if [ -n "$avg_texture" ] && [ "$avg_texture" != "0" ] && [ "$avg_baseline" != "0" ]; then
        speedup_texture=$(echo "scale=3; $avg_baseline / $avg_texture" | bc -l)
    else
        speedup_texture="N/A"
    fi
    
    if [ -n "$avg_hybrid" ] && [ "$avg_hybrid" != "0" ] && [ "$avg_baseline" != "0" ]; then
        speedup_hybrid=$(echo "scale=3; $avg_baseline / $avg_hybrid" | bc -l)
    else
        speedup_hybrid="N/A"
    fi
    
    # Store results
    results["${dataset}_dim"]=$DIM
    results["${dataset}_baseline"]=$avg_baseline
    results["${dataset}_shared"]=$avg_shared
    results["${dataset}_texture"]=$avg_texture
    results["${dataset}_hybrid"]=$avg_hybrid
    results["${dataset}_cpu"]=$avg_cpu
    results["${dataset}_speedup_shared"]=$speedup_shared
    results["${dataset}_speedup_texture"]=$speedup_texture
    results["${dataset}_speedup_hybrid"]=$speedup_hybrid
    
    echo "" | tee -a "$LOG_FILE"
    echo "Average Results:" | tee -a "$LOG_FILE"
    echo "  Baseline (No Shared):  ${avg_baseline}s" | tee -a "$LOG_FILE"
    echo "  Shared Memory:         ${avg_shared}s  (speedup: ${speedup_shared}x over baseline)" | tee -a "$LOG_FILE"
    echo "  Texture Memory:        ${avg_texture}s  (speedup: ${speedup_texture}x over baseline)" | tee -a "$LOG_FILE"
    echo "  Texture + Shared:      ${avg_hybrid}s  (speedup: ${speedup_hybrid}x over baseline)" | tee -a "$LOG_FILE"
    echo "  CPU:                   ${avg_cpu}s" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
done

# Print summary table
echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "SUMMARY TABLE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
printf "%-20s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-8s | %-8s | %-8s\n" \
    "Dataset" "Dimensions" "Baseline(s)" "Shared(s)" "Texture(s)" "Hybrid(s)" "CPU(s)" "Spd-Sh" "Spd-Tx" "Spd-Hy" | tee -a "$LOG_FILE"
printf "%s\n" "$(printf '=%.0s' {1..140})" | tee -a "$LOG_FILE"

for dataset in "${DATASETS[@]}"; do
    dim=${results["${dataset}_dim"]}
    if [ -z "$dim" ]; then
        continue  # Skip if dataset wasn't processed
    fi
    
    baseline=${results["${dataset}_baseline"]}
    shared=${results["${dataset}_shared"]}
    texture=${results["${dataset}_texture"]}
    hybrid=${results["${dataset}_hybrid"]}
    cpu=${results["${dataset}_cpu"]}
    speedup_shared=${results["${dataset}_speedup_shared"]}
    speedup_texture=${results["${dataset}_speedup_texture"]}
    speedup_hybrid=${results["${dataset}_speedup_hybrid"]}
    
    printf "%-20s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-8s | %-8s | %-8s\n" \
        "$dataset" "${dim}x${dim}" "$baseline" "$shared" "$texture" "$hybrid" "$cpu" "$speedup_shared" "$speedup_texture" "$speedup_hybrid" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Benchmark complete!" | tee -a "$LOG_FILE"
echo "Results saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
