#ifndef DGEMM_2D_TILED_DRIVER_H
#define DGEMM_2D_TILED_DRIVER_H

#ifdef __cplusplus
extern "C" {
#endif

bool dgemm_2d_tiled_driver(float alpha, float beta, int M, int N, int K, float* hA, float* hB, float* hC);

#ifdef __cplusplus
}
#endif

#endif // DGEMM_2D_TILED_DRIVER_H