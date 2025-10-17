#ifndef DGEMM_BANK_CONFLICTS_DRIVER_H
#define DGEMM_BANK_CONFLICTS_DRIVER_H

#ifdef __cplusplus
extern "C" {
#endif

bool dgemm_bank_conflicts_driver(double alpha, double beta, int M, int N, int K, double* hA, double* hB, double* hC);

#ifdef __cplusplus
}
#endif

#endif // DGEMM_BANK_CONFLICTS_DRIVER_H