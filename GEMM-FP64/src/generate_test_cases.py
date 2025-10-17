import numpy as np
import os

def generate_matrix_multiplication_test_cases(test_cases, output_dir="test_cases"):
    """
    Generates pairs of matrices (A, B) for multiplication, computes their product (C),
    and saves all three as CSV files.

    Args:
        test_cases (list of tuples): A list where each tuple contains the dimensions
                                     and type for a test case in the format
                                     (rows_A, cols_A, cols_B, 'type').
        output_dir (str): The directory where the CSV files will be saved.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    for i, (rows_A, cols_A, cols_B, case_type) in enumerate(test_cases):
        case_num = i + 1
        print(f"\n--- Generating Test Case {case_num} ({case_type}) ---")
        print(f"  Matrix A dimensions: {rows_A}x{cols_A}")
        print(f"  Matrix B dimensions: {cols_A}x{cols_B}")

        # The number of rows in B must equal the number of columns in A
        rows_B = cols_A

        # --- Generate Matrices based on case_type ---

        # Matrix A is always random for these test cases
        matrix_A = np.random.randint(-10, 11, size=(rows_A, cols_A))

        # Matrix B's generation depends on the specified type
        if case_type == 'random':
            matrix_B = np.random.randint(-10, 11, size=(rows_B, cols_B))
        elif case_type == 'identity':
            # An identity matrix must be square.
            if rows_B != cols_B:
                print(f"  ERROR: Cannot create a non-square identity matrix for B ({rows_B}x{cols_B}). Skipping.")
                continue
            matrix_B = np.identity(rows_B, dtype=int)
        else:
            print(f"  ERROR: Unknown case type '{case_type}'. Skipping.")
            continue

        # Perform the matrix multiplication
        matrix_C = np.dot(matrix_A, matrix_B)

        print(f"  Result C dimensions: {matrix_C.shape[0]}x{matrix_C.shape[1]}")

        # Define file paths
        path_A = os.path.join(output_dir, f"A_{case_num}.csv")
        path_B = os.path.join(output_dir, f"B_{case_num}.csv")
        path_C = os.path.join(output_dir, f"AB_{case_num}.csv")

        # Save the matrices to CSV files
        # fmt='%d' ensures the numbers are saved as integers
        np.savetxt(path_A, matrix_A, delimiter=",", fmt='%d')
        np.savetxt(path_B, matrix_B, delimiter=",", fmt='%d')
        np.savetxt(path_C, matrix_C, delimiter=",", fmt='%d')

        print(f"  Successfully saved: {path_A}, {path_B}, {path_C}")


if __name__ == "__main__":
    # --- Define your test cases here ---
    # Each tuple is (rows_A, cols_A, cols_B, case_type)
    # Supported types: 'random', 'identity'
    # The dimensions of B's rows are inferred from A's columns.
    test_definitions = [
        # (16, 16, 16, "random"),
        # (32, 32, 32, "random"),
        # (64, 64, 64, "random"),
        (1024, 1024, 1024, "random"),
        (128, 128, 128, "random"),
        (256, 128, 256, "random"),
        # (64, 128, 64, "random"),
        # (128, 256, 64, "random"),
        # (8, 16, 4, "random"),
        # (8, 16, 8, "random")
    ]

    generate_matrix_multiplication_test_cases(test_definitions)
