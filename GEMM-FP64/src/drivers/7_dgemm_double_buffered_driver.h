#ifndef DGEMM_DOUBLE_BUFFERED_DRIVER_H
#define DGEMM_DOUBLE_BUFFERED_DRIVER_H

#ifdef __cplusplus
extern "C" {
#endif

bool dgemm_double_buffered_driver(float alpha, float beta, int M, int N, int K, float* hA, float* hB, float* hC, bool debug);

#ifdef __cplusplus
}
#endif

#endif // DGEMM_DOUBLE_BUFFERED_DRIVER_H