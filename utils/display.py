import numpy as np

def print_table(matrix):
    """
    Exibe uma matriz como uma tabela formatada.

    Parâmetros:
    - matrix (numpy.ndarray): A matriz a ser exibida.
    """
    num_rows, num_cols = matrix.shape
    max_width = len(str(num_rows * num_cols))

    print(f"    ".rjust(max_width + 3), end=" |")
    for col_idx in range(num_cols):
        print(f" Col{col_idx} ".center(10), end=" |")
    print()

    print(f"    ".rjust(max_width + 5) + "-" * (max_width + 1 + 11 * num_cols))

    for row_idx in range(num_rows):
        print(f" Row{row_idx}".ljust(max_width + 1), end=" |")
        for col_idx in range(num_cols):
            print(f" {matrix[row_idx, col_idx]} ".center(10), end=" |")
        print()
