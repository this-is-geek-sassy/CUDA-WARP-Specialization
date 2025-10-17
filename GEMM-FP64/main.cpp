#include <sstream>
#include <fstream>
#include <iostream>
#include "drivers/dgemm_basic_driver.h"

/// @brief Used to store Matrix on the host.
/// @param m Number of rows of the matrix.
/// @param n Number of columns of the matrix.
/// @param ptr Pointer to the first element of the matrix. (Stored in Row-Major order)
struct Matrix {
  int m;
  int n;
  double* ptr;
};

/// @brief Loads the matrix stored in <fileName>.csv
/// @param fileName The path to a .csv file containing the matrix
/// @return Returns an instance of Matrix
Matrix inputMatrix(std::string fileName) {
  // Open the file.
  std::ifstream file(fileName);

  // Return a NULL Matrix if unable to open file.
  if(!file) {
    std::cerr << "ERROR: Unable to open file: " << fileName << std::endl;
    return Matrix{0, 0, NULL};
  }

  // PHASE 1: Count the number of rows (m) and columns (n) in the matrix.
  int n = 0, m = 0, lastN = -1;
  std::string line;

  //// Read the file line by line.
  while(std::getline(file, line)) {
    //// Every line corresponds to a new row in the matrix.
    m++;

    //// Count the number of elements in each row.
    std::stringstream stream(line);
    std::string item;
    n = 0;
    while(std::getline(stream, item, ','))
      n++;
    
    //// Compare the numbers of elements in the current row with the previous one.
    //// lastN != -1 skips the comparision if it's the first row.
    if(lastN != -1 && n != lastN) {
      std::cerr << "ERROR: Cannot have different number of elements in any two rows" << std::endl;
      return Matrix{0, 0, NULL};
    }
    lastN = n;
  }

  // PHASE 2: We store the matrix in memory.
  //// Close and reopen the file.
  file.close();
  file.open(fileName);

  //// Allocate space in memory for the matrix.
  double* arr = (double*) malloc(m * n * sizeof(double));

  //// Keep track of the current row and column in i and j, respectively.
  int i = 0, j = 0;

  //// Read the file line by line.
  while(std::getline(file, line)) {
    std::stringstream stream(line);
    std::string item;

    //// Read the line element by element.
    j = 0;
    while(std::getline(stream, item, ',')) {
      arr[i * n + j] = std::stof(item);
      j++;
    }
    
    //// Increment i before we read the next row.
    i++;
  }

  return Matrix{m, n, arr};
}

/// @brief Verifies whether two martrices have identical elements.
/// @param M Number of rows of the matrices.
/// @param N Number of columns of the matrices.
/// @param hP Pointer to the first matrix.
/// @param hC Pointer to the second matrix.
bool verify(int M, int N, double* hP, double* hC) {
  bool flag = false;
  for(int i = 0; i < M; i++) {
    for(int j = 0; j < N; j++) {
      if(hC[i * N + j] != hP[i * N + j]) {
        flag = true;
        break;
      }
    }
    if(flag) break;
  }

  if(flag) std::cerr << "ERROR: Incorrect output!" << std::endl;
  else std::cout << "Correct output!" << std::endl;
  return flag;
}

/// @brief Prints a matrix to std::cout.
/// @param M Number of rows.
/// @param N Number of columns.
/// @param A Pointer to the matrix.
void print(int M, int N, double* A) {
  for(int i = 0; i < M; i++) {
    for(int j = 0; j < N; j++) {
      std::cout << A[i * N + j] << " ";
    }
    std::cout << std::endl;
  }
}

int main(int argc, char* argv[]) {
  // Throw error on incorrect usage.
  if(argc < 3) {
    std::cerr << "USAGE: <TEST_NO> <KERNEL_NO>" << std::endl;
    return 1;
  }

  // Parse test case number and generate filenames.
  int test_case_no = std::stoi(argv[1]);
  std::string AFilename = "./test_cases/A_" + std::to_string(test_case_no) + ".csv";
  std::string BFilename = "./test_cases/B_" + std::to_string(test_case_no) + ".csv";
  std::string ABFilename = "./test_cases/AB_" + std::to_string(test_case_no) + ".csv";

  // Parse kernel no.
  int kernel_no = std::stoi(argv[2]);

  // Input matrix A.
  Matrix hA = inputMatrix(AFilename);
  if(hA.ptr == NULL) {
    std::cerr << "ERROR: Unable to load Matrix A" << std::endl;
    return 1;
  }

  // Input matrix B.
  Matrix hB = inputMatrix(BFilename);
  if(hB.ptr == NULL) {
    std::cerr << "ERROR: Unable to load Matrix B" << std::endl;
    return 1;
  }

  // Input matrix P.
  Matrix hP = inputMatrix(ABFilename);
  if(hP.ptr == NULL) {
    std::cerr << "ERROR: Unable to load Matrix P" << std::endl;
    return 1;
  }

  // Verify if A and B can be multiplied together.
  if(hA.n != hB.m) {
    std::cerr << "ERROR: Matrices A and B cannot be multiplied together" << std::endl;
    return 1;
  }

  int M = hA.m;
  int K = hA.n;
  int N = hB.n;

  // Allocate memory for product matrix.
  double* hC = (double*) malloc(M * N * sizeof(double));

  // Call the requested kernel.
  bool success = true;
  switch(kernel_no) {
    case 1:
      success = dgemm_basic_driver(M, N, K, hA.ptr, hB.ptr, hC);
      if(success) verify(M, N, hP.ptr, hC);
      break;
    default:
      std::cerr << "ERROR: Invalid kernel requested!" << std::endl;
      break;
  }
 
  // Free pointers.
  free(hA.ptr);
  free(hB.ptr);
  free(hP.ptr);
  free(hC);
    
  return success;
}
