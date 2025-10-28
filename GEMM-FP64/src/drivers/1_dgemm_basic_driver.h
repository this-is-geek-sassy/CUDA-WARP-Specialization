#ifndef DGEMM_BASIC_DRIVER_H
#define DGEMM_BASIC_DRIVER_H

#ifdef __cplusplus
extern "C" {
#endif

bool dgemm_basic_driver(float alpha, float beta, int M, int N, int K, float* hA, float* hB, float* hC);

#ifdef __cplusplus
}
#endif

#endif // DGEMM_BASIC_DRIVER_H