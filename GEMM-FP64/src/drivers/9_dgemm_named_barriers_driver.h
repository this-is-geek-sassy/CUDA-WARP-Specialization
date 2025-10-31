#ifndef DGEMM_NAMED_BARRIERS_DRIVER_H
#define DGEMM_NAMED_BARRIERS_DRIVER_H

#ifdef __cplusplus
extern "C" {
#endif

bool dgemm_named_barriers_driver(float alpha, float beta, int M, int N, int K, float* hA, float* hB, float* hC);

#ifdef __cplusplus
}
#endif

#endif // DGEMM_NAMED_BARRIERS_DRIVER_H