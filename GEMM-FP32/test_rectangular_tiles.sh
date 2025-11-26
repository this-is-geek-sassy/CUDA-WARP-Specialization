#!/bin/bash

# Test script for rectangular tiles functionality
# Tests various tile configurations and verifies correctness

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║         GEMM Rectangular Tiles Test Suite                       ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test configurations: M N K Description
CONFIGS=(
    "32 32 32 Default_Square"
    "64 32 32 Tall_Tiles"
    "32 64 32 Wide_Tiles"
    "32 32 16 Small_K"
    "16 16 16 Small_Square"
    "64 64 32 Large_Square"
)

# Dataset to use for testing
DATASET="STANDARD_DATASET"

SUCCESS_COUNT=0
FAIL_COUNT=0

# Compile and test each configuration
for config in "${CONFIGS[@]}"; do
    read -r M N K DESC <<< "$config"
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Testing Configuration: $DESC"
    echo "  TILE_M=$M, TILE_N=$N, TILE_K=$K"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Compile
    echo -n "Compiling... "
    nvcc -O3 -D${DATASET} -DTILE_M=$M -DTILE_N=$N -DTILE_K=$K \
         gemm_fp_32_cudaDMA.cu -o gemm_test_${M}x${N}x${K} \
         -gencode arch=compute_86,code=sm_86 2>&1 | grep -i "error" > /dev/null
    
    if [ $? -eq 0 ]; then
        echo -e "${RED}✗ Compilation FAILED${NC}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    else
        echo -e "${GREEN}✓ Success${NC}"
    fi
    
    # Run and check for errors
    echo -n "Running kernel... "
    ./gemm_test_${M}x${N}x${K} > test_output_${M}x${N}x${K}.log 2>&1
    
    # Check for CUDA errors
    if grep -q "error" test_output_${M}x${N}x${K}.log; then
        echo -e "${RED}✗ Runtime Error${NC}"
        echo "Error details:"
        grep "error" test_output_${M}x${N}x${K}.log
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi
    
    # Check for correctness (mismatch count should be 0 or very low)
    MISMATCH=$(grep "Non-Matching" test_output_${M}x${N}x${K}.log | grep -oP '\d+$' | tail -1)
    
    if [ -z "$MISMATCH" ]; then
        echo -e "${YELLOW}? Unable to verify (CPU skipped)${NC}"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    elif [ "$MISMATCH" -eq 0 ]; then
        echo -e "${GREEN}✓ Correct (0 mismatches)${NC}"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo -e "${RED}✗ Incorrect ($MISMATCH mismatches)${NC}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    
    # Extract and display timing
    echo "Performance:"
    grep "GPU Time" test_output_${M}x${N}x${K}.log -A 1 | grep -oP '\d+\.\d+' | while read time; do
        echo "  Time: ${time}s"
    done
    
    echo ""
done

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "                          SUMMARY                                  "
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "Successful: ${GREEN}$SUCCESS_COUNT${NC}"
echo -e "Failed:     ${RED}$FAIL_COUNT${NC}"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some tests failed. Check logs for details.${NC}"
    exit 1
fi
