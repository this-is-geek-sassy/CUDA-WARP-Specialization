#ifndef DGEMM_REGISTER_TILED_DRIVER_H
#define DGEMM_REGISTER_TILED_DRIVER_H

#ifdef __cplusplus
extern "C" {
#endif

bool dgemm_register_tiled_driver(float alpha, float beta, int M, int N, int K, float* hA, float* hB, float* hC, bool debug);

#ifdef __cplusplus
}
#endif

#endif // DGEMM_REGISTER_TILED_DRIVER_H