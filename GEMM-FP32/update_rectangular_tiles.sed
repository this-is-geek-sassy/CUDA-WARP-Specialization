# Update cudaDMA single buffering kernel
s/__shared__ fp32_t As\[TILE_SIZE\]\[TILE_SIZE\];/__shared__ fp32_t As[TILE_M][TILE_K];/g
s/__shared__ fp32_t Bs\[TILE_SIZE\]\[TILE_SIZE\];/__shared__ fp32_t Bs[TILE_K][TILE_N];/g
s/cudaDMAStrided<true, 16, 128, DMA_THREADS_PER_LD, TILE_SIZE>/cudaDMAStrided<true, 16, TILE_K * sizeof(fp32_t), DMA_THREADS_PER_LD, TILE_M>/g
s/TILE_SIZE \* sizeof(fp32_t));/TILE_K * sizeof(fp32_t));/g
s/(nk + TILE_SIZE - 1) \/ TILE_SIZE/(nk + TILE_K - 1) \/ TILE_K/g
s/(TILE_SIZE \* TILE_SIZE) \/ COMPUTE_THREADS_PER_CTA; \/\/ 4/(TILE_M * TILE_N) \/ COMPUTE_THREADS_PER_CTA/g
s/fp32_t sums\[4\] = {0.0f, 0.0f, 0.0f, 0.0f};/fp32_t sums[elements_per_thread];\n        for (int i = 0; i < elements_per_thread; i++) sums[i] = 0.0f;/g
s/linear_idx \/ TILE_SIZE;/linear_idx \/ TILE_N;/g
s/linear_idx % TILE_SIZE;/linear_idx % TILE_N;/g
s/by \* TILE_SIZE + ty;/by * TILE_M + ty;/g
s/bx \* TILE_SIZE + tx;/bx * TILE_N + tx;/g
s/t \* TILE_SIZE;/t * TILE_K;/g
s/for (int k = 0; k < TILE_SIZE; k++)/for (int k = 0; k < TILE_K; k++)/g
s/(NJ + TILE_SIZE - 1) \/ TILE_SIZE/(NJ + TILE_N - 1) \/ TILE_N/g
s/(NI + TILE_SIZE - 1) \/ TILE_SIZE/(NI + TILE_M - 1) \/ TILE_M/g
s/dim3 block(TILE_SIZE, TILE_SIZE);/dim3 block(TILE_N, TILE_M);/g
