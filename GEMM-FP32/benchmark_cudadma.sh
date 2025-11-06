#!/bin/bash

# Benchmark script for gemm_fp_32_cudaDMA across all dataset sizes
# Builds each dataset size, runs 5 times, and computes average GPU time

# Dataset sizes to benchmark
DATASETS=(
    "MINI_DATASET"
    "SMALL_DATASET"
    "STANDARD_DATASET"
    "LARGE_DATASET"
    "EXTRALARGE_DATASET"
    "HUGE_DATASET"
    "HUMONGOUS_DATASET"
)

# Corresponding dimensions for reference
DIMENSIONS=(
    "32x32"
    "124x124"
    "512x512"
    "1024x1024"
    "2048x2048"
    "4096x4096"
    "8192x8192"
)

ITERATIONS=5
MAKEFILE="Makefile_dma"
EXECUTABLE="gemm_fp_32_cudadma"

# Create timestamped log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="benchmark_results_${TIMESTAMP}.txt"

# Function to log and print simultaneously
log_print() {
    echo "$1" | tee -a "$LOGFILE"
}

log_print "========================================"
log_print "cudaDMA GEMM Benchmark Suite"
log_print "========================================"
log_print "Date: $(date)"
log_print "Iterations per dataset: $ITERATIONS"
log_print "Log file: $LOGFILE"
log_print ""

# Results array
declare -a RESULTS

for i in "${!DATASETS[@]}"; do
    DATASET="${DATASETS[$i]}"
    DIM="${DIMENSIONS[$i]}"
    
    log_print "----------------------------------------"
    log_print "Dataset: $DATASET ($DIM)"
    log_print "----------------------------------------"
    
    # Clean previous build
    make -f "$MAKEFILE" clean > /dev/null 2>&1
    
    # Build with current dataset size
    log_print "Building..."
    sed -i.bak "s/^DATASET :=.*/DATASET := -D$DATASET/" "$MAKEFILE"
    
    if ! make -f "$MAKEFILE" > /dev/null 2>&1; then
        log_print "ERROR: Build failed for $DATASET"
        # Restore original makefile
        mv "${MAKEFILE}.bak" "$MAKEFILE" 2>/dev/null
        continue
    fi
    
    # Restore original makefile
    mv "${MAKEFILE}.bak" "$MAKEFILE" 2>/dev/null
    
    log_print "Running $ITERATIONS iterations..."
    
    # Array to store times
    declare -a times_baseline
    declare -a times_cudadma
    declare -a times_cpu
    
    for run in $(seq 1 $ITERATIONS); do
        echo -n "  Run $run/$ITERATIONS... "
        echo -n "  Run $run/$ITERATIONS... " >> "$LOGFILE"
        
        # Run the executable and capture output
        output=$(./"$EXECUTABLE" 2>&1)
        
        # Extract GPU times (looking for lines after "GPU Time in seconds")
        # Parse baseline FP32 time
        baseline_time=$(echo "$output" | grep -A 1 "GPU Time in seconds (FP32):" | grep -E "^[0-9]+\.[0-9]+" | head -1)
        
        # Parse cudaDMA time
        cudadma_time=$(echo "$output" | grep -A 1 "GPU Time in seconds (FP32 with cudaDMA):" | grep -E "^[0-9]+\.[0-9]+" | head -1)
        
        # Parse CPU time
        cpu_time=$(echo "$output" | grep -A 1 "CPU Time in seconds:" | grep -E "^[0-9]+\.[0-9]+" | head -1)
        
        if [ -z "$baseline_time" ] || [ -z "$cudadma_time" ]; then
            log_print "FAILED (could not parse timing)"
            continue
        fi
        
        times_baseline+=("$baseline_time")
        times_cudadma+=("$cudadma_time")
        if [ -n "$cpu_time" ]; then
            times_cpu+=("$cpu_time")
        fi
        
        log_print "Baseline: ${baseline_time}s, cudaDMA: ${cudadma_time}s, CPU: ${cpu_time}s"
    done
    
    # Calculate averages
    if [ ${#times_baseline[@]} -eq 0 ]; then
        log_print "ERROR: No successful runs for $DATASET"
        continue
    fi
    
    avg_baseline=$(awk 'BEGIN {sum=0} {sum+=$1} END {print sum/NR}' <<< "$(printf '%s\n' "${times_baseline[@]}")")
    avg_cudadma=$(awk 'BEGIN {sum=0} {sum+=$1} END {print sum/NR}' <<< "$(printf '%s\n' "${times_cudadma[@]}")")
    
    # Calculate CPU average if available
    if [ ${#times_cpu[@]} -gt 0 ]; then
        avg_cpu=$(awk 'BEGIN {sum=0} {sum+=$1} END {print sum/NR}' <<< "$(printf '%s\n' "${times_cpu[@]}")")
    else
        avg_cpu="N/A"
    fi
    
    speedup=$(awk "BEGIN {printf \"%.3f\", $avg_baseline / $avg_cudadma}")
    
    log_print ""
    log_print "Average Baseline:  ${avg_baseline} seconds"
    log_print "Average cudaDMA:   ${avg_cudadma} seconds"
    log_print "Average CPU:       ${avg_cpu} seconds"
    log_print "Speedup:           ${speedup}x"
    log_print ""
    
    # Store results
    RESULTS+=("$DATASET|$DIM|$avg_baseline|$avg_cudadma|$avg_cpu|$speedup")
    
    # Cleanup
    unset times_baseline
    unset times_cudadma
    unset times_cpu
done

# Print summary table
log_print "========================================"
log_print "SUMMARY"
log_print "========================================"
{
    printf "%-20s %-15s %-15s %-15s %-15s %-10s\n" "Dataset" "Dimensions" "Baseline (s)" "cudaDMA (s)" "CPU (s)" "Speedup"
    echo "--------------------------------------------------------------------------------------------"
    
    for result in "${RESULTS[@]}"; do
        IFS='|' read -r dataset dim baseline cudadma cpu speedup <<< "$result"
        printf "%-20s %-15s %-15s %-15s %-15s %-10s\n" "$dataset" "$dim" "$baseline" "$cudadma" "$cpu" "${speedup}x"
    done
} | tee -a "$LOGFILE"

log_print "========================================"
log_print "Benchmark complete!"
log_print "Results saved to: $LOGFILE"
log_print "========================================"
