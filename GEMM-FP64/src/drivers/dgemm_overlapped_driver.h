#ifndef DGEMM_OVERLAPPED_DRIVER_H
#define DGEMM_OVERLAPPED_DRIVER_H

#ifdef __cplusplus
extern "C" {
#endif

bool dgemm_overlapped_driver(double alpha, double beta, int M, int N, int K, double* hA, double* hB, double* hC);

#ifdef __cplusplus
}
#endif

#endif // DGEMM_OVERLAPPED_DRIVER_H