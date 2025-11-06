#ifndef DGEMM_DOUBLE_BUFFERED_CPASYNC_DRIVER_H
#define DGEMM_DOUBLE_BUFFERED_CPASYNC_DRIVER_H

#ifdef __cplusplus
extern "C" {
#endif

bool dgemm_double_buffered_cpasync_driver(float alpha, float beta, int M, int N, int K, float* hA, float* hB, float* hC);

#ifdef __cplusplus
}
#endif

#endif // DGEMM_DOUBLE_BUFFERED_CPASYNC_DRIVER_H