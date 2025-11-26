#ifndef DGEMM_BANK_CONFLICTS_DRIVER_H
#define DGEMM_BANK_CONFLICTS_DRIVER_H

#ifdef __cplusplus
extern "C" {
#endif

bool dgemm_bank_conflicts_driver(float alpha, float beta, int M, int N, int K, float* hA, float* hB, float* hC, bool debug);

#ifdef __cplusplus
}
#endif

#endif // DGEMM_BANK_CONFLICTS_DRIVER_H