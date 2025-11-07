#!/bin/bash

# Benchmark script for gemm_fp_32_cudaDMA across all dataset sizes
# Builds each dataset size, runs 5 times, and computes average GPU time
# Usage: ./benchmark_cudadma.sh [max_dimension]
# Example: ./benchmark_cudadma.sh 1024  (runs benchmarks up to 1024x1024)

# Dataset sizes to benchmark
ALL_DATASETS=(
    "MINI_DATASET"
    "SMALL_DATASET"
    "STANDARD_DATASET"
    "LARGE_DATASET"
    "EXTRALARGE_DATASET"
    "HUGE_DATASET"
    "HUMONGOUS_DATASET"
)

# Corresponding dimensions for reference
ALL_DIMENSIONS=(
    "32x32"
    "124x124"
    "512x512"
    "1024x1024"
    "2048x2048"
    "4096x4096"
    "8192x8192"
)

# Numeric dimension values for comparison
DIMENSION_VALUES=(
    32
    124
    512
    1024
    2048
    4096
    8192
)

# Parse command line argument for max dimension
MAX_DIM=${1:-8192}  # Default to largest if not specified

# Filter datasets based on max dimension
DATASETS=()
DIMENSIONS=()
for i in "${!ALL_DATASETS[@]}"; do
    if [ "${DIMENSION_VALUES[$i]}" -le "$MAX_DIM" ]; then
        DATASETS+=("${ALL_DATASETS[$i]}")
        DIMENSIONS+=("${ALL_DIMENSIONS[$i]}")
    fi
done

# Check if any datasets match
if [ ${#DATASETS[@]} -eq 0 ]; then
    echo "ERROR: No datasets found for max dimension $MAX_DIM"
    echo "Valid dimensions are: ${DIMENSION_VALUES[*]}"
    exit 1
fi

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
log_print "Max dimension: ${MAX_DIM}x${MAX_DIM}"
log_print "Datasets to run: ${#DATASETS[@]} (${DATASETS[*]})"
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
    declare -a times_cudadma_single
    declare -a times_cudadma_double
    declare -a times_cpu
    
    for run in $(seq 1 $ITERATIONS); do
        echo -n "  Run $run/$ITERATIONS... "
        echo -n "  Run $run/$ITERATIONS... " >> "$LOGFILE"
        
        # Run the executable and capture output
        output=$(./"$EXECUTABLE" 2>&1)
        
        # Extract GPU times (looking for lines after "GPU Time in seconds")
        # Parse baseline FP32 time
        baseline_time=$(echo "$output" | grep -A 1 "GPU Time in seconds (FP32):" | grep -E "^[0-9]+\.[0-9]+" | head -1)
        
        # Parse cudaDMA single-buffer time
        cudadma_single_time=$(echo "$output" | grep -A 1 "GPU Time in seconds (FP32 with cudaDMA Single-Buffer):" | grep -E "^[0-9]+\.[0-9]+" | head -1)
        
        # Parse cudaDMA double-buffer time
        cudadma_double_time=$(echo "$output" | grep -A 1 "GPU Time in seconds (FP32 with cudaDMA Double-Buffer):" | grep -E "^[0-9]+\.[0-9]+" | head -1)
        
        # Parse CPU time
        cpu_time=$(echo "$output" | grep -A 1 "CPU Time in seconds:" | grep -E "^[0-9]+\.[0-9]+" | head -1)
        
        if [ -z "$baseline_time" ] || [ -z "$cudadma_single_time" ] || [ -z "$cudadma_double_time" ]; then
            log_print "FAILED (could not parse timing)"
            continue
        fi
        
        times_baseline+=("$baseline_time")
        times_cudadma_single+=("$cudadma_single_time")
        times_cudadma_double+=("$cudadma_double_time")
        if [ -n "$cpu_time" ]; then
            times_cpu+=("$cpu_time")
        fi
        
        log_print "Baseline: ${baseline_time}s, Single: ${cudadma_single_time}s, Double: ${cudadma_double_time}s, CPU: ${cpu_time}s"
    done
    
    # Calculate averages
    if [ ${#times_baseline[@]} -eq 0 ]; then
        log_print "ERROR: No successful runs for $DATASET"
        continue
    fi
    
    avg_baseline=$(awk 'BEGIN {sum=0} {sum+=$1} END {print sum/NR}' <<< "$(printf '%s\n' "${times_baseline[@]}")")
    avg_cudadma_single=$(awk 'BEGIN {sum=0} {sum+=$1} END {print sum/NR}' <<< "$(printf '%s\n' "${times_cudadma_single[@]}")")
    avg_cudadma_double=$(awk 'BEGIN {sum=0} {sum+=$1} END {print sum/NR}' <<< "$(printf '%s\n' "${times_cudadma_double[@]}")")
    
    # Calculate CPU average if available
    if [ ${#times_cpu[@]} -gt 0 ]; then
        avg_cpu=$(awk 'BEGIN {sum=0} {sum+=$1} END {print sum/NR}' <<< "$(printf '%s\n' "${times_cpu[@]}")")
    else
        avg_cpu="N/A"
    fi
    
    speedup_single=$(awk "BEGIN {printf \"%.3f\", $avg_baseline / $avg_cudadma_single}")
    speedup_double=$(awk "BEGIN {printf \"%.3f\", $avg_baseline / $avg_cudadma_double}")
    
    log_print ""
    log_print "Average Baseline:       ${avg_baseline} seconds"
    log_print "Average cudaDMA Single: ${avg_cudadma_single} seconds"
    log_print "Average cudaDMA Double: ${avg_cudadma_double} seconds"
    log_print "Average CPU:            ${avg_cpu} seconds"
    log_print "Speedup (Single):       ${speedup_single}x"
    log_print "Speedup (Double):       ${speedup_double}x"
    log_print ""
    
    # Store results
    RESULTS+=("$DATASET|$DIM|$avg_baseline|$avg_cudadma_single|$avg_cudadma_double|$avg_cpu|$speedup_single|$speedup_double")
    
    # Cleanup
    unset times_baseline
    unset times_cudadma_single
    unset times_cudadma_double
    unset times_cpu
done

# Print summary table
log_print "========================================"
log_print "SUMMARY"
log_print "========================================"
{
    printf "%-20s %-12s %-12s %-12s %-12s %-12s %-10s %-10s\n" "Dataset" "Dimensions" "Baseline(s)" "Single(s)" "Double(s)" "CPU(s)" "Spd-S" "Spd-D"
    echo "----------------------------------------------------------------------------------------------------------------------------"
    
    for result in "${RESULTS[@]}"; do
        IFS='|' read -r dataset dim baseline single double cpu speedup_s speedup_d <<< "$result"
        printf "%-20s %-12s %-12s %-12s %-12s %-12s %-10s %-10s\n" "$dataset" "$dim" "$baseline" "$single" "$double" "$cpu" "${speedup_s}x" "${speedup_d}x"
    done
} | tee -a "$LOGFILE"

log_print "========================================"
log_print "Benchmark complete!"
log_print "Results saved to: $LOGFILE"
log_print "========================================"
log_print ""
log_print "Usage: $0 [max_dimension]"
log_print "Available dimensions: ${DIMENSION_VALUES[*]}"
log_print "Example: $0 1024  (benchmarks up to 1024x1024)"

