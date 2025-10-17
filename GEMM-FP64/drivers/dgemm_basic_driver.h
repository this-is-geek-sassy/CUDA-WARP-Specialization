#ifndef DGEMM_BASIC_DRIVER_H
#define DGEMM_BASIC_DRIVER_H

#ifdef __cplusplus
extern "C" {
#endif

void dgemm_basic_driver(int M, int N, int K, double* hA, double* hB, double* hC);

#ifdef __cplusplus
}
#endif

#endif // DGEMM_BASIC_DRIVER_H