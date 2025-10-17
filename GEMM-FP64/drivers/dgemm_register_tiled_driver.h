#ifndef DGEMM_REGISTER_TILED_DRIVER_H
#define DGEMM_REGISTER_TILED_DRIVER_H

#ifdef __cplusplus
extern "C" {
#endif

bool dgemm_register_tiled_driver(int M, int N, int K, double* hA, double* hB, double* hC);

#ifdef __cplusplus
}
#endif

#endif // DGEMM_REGISTER_TILED_DRIVER_H