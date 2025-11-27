#!/bin/bash

# ================= CONFIGURATION =================
BINARY_PATH="./bin/dgemm"
KERNELS=(2)
TEST_CASES=(3 4)
# =================================================

# Check if binary exists
if [ ! -f "$BINARY_PATH" ]; then
    echo "Error: Executable '$BINARY_PATH' not found."
    exit 1
fi

# Print Table Header
printf "%-12s %-10s %-15s\n" "Test Case" "Kernel" "Avg Time"
printf "%s\n" "----------------------------------------"

# Loop through Test Cases
for tc in "${TEST_CASES[@]}"; do
    
    # Loop through Kernels
    for k in "${KERNELS[@]}"; do
        
        # --- 1. Warmup Run ---
        # We redirect output to /dev/null to hide it
        "$BINARY_PATH" "$k" "$tc" > /dev/null 2>&1
        
        total_time=0
        
        # --- 2. Actual Runs (3 times) ---
        for i in {1..3}; do
            # Run and capture output
            output=$("$BINARY_PATH" "$tc" "$k")
            
            # Extract the LAST floating point number from the output
            # 1. grep -oE extracts all numbers (integers or floats)
            # 2. tail -n 1 takes the very last one found
            exec_time=$(echo "$output" | grep -oE '[0-9]*\.[0-9]+|[0-9]+' | tail -n 1)
            
            # If no number was found, default to 0 to prevent syntax errors
            if [ -z "$exec_time" ]; then
                exec_time=0
            fi
            
            # Accumulate sum (using awk for float addition)
            total_time=$(awk "BEGIN {print $total_time + $exec_time}")
        done
        
        # --- 3. Calculate Average ---
        avg_time=$(awk "BEGIN {print $total_time / 3}")
        
        # Print row
        printf "%-12s %-10s %-15.4f\n" "$tc" "$k" "$avg_time"
        
    done
done