#ifndef DGEMM_WARP_SPECIALIZED_CPASYNC_DRIVER_H
#define DGEMM_WARP_SPECIALIZED_CPASYNC_DRIVER_H

#ifdef __cplusplus
extern "C" {
#endif

bool dgemm_warp_specialized_cpasync_driver(float alpha, float beta, int M, int N, int K, float* hA, float* hB, float* hC, bool debug);

#ifdef __cplusplus
}
#endif

#endif // DGEMM_WARP_SPECIALIZED_CPASYNC_DRIVER_H