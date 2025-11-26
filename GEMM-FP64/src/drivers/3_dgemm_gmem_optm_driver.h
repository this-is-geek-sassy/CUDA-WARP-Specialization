#ifndef DGEMM_GMEM_OPTM_DRIVER_H
#define DGEMM_GMEM_OPTM_DRIVER_H

#ifdef __cplusplus
extern "C" {
#endif

bool dgemm_gmem_optm_driver(float alpha, float beta, int M, int N, int K, float* hA, float* hB, float* hC, bool debug);

#ifdef __cplusplus
}
#endif

#endif // DGEMM_GMEM_OPTM_DRIVER_H