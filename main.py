from typing import Dict, List, Tuple, Union, Any
import numpy as np

def generate_random_floats_with_mean_variance(mean: float = 0.0, variance: float = 0.0) -> List[float]:
    return ((np.array(variance) * np.random.randn( 10 ** 6 )) + np.array(mean)) .tolist()

def calculate_eigen(matrix: np.ndarray) -> tuple[Any, Any]:
    # Check if matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square")
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    return eigenvalues, eigenvectors

def create_a_matrix(cx: np.ndarray) -> np.ndarray:
    unnormalized_eigenvectors = None
    eigenvalues, eigenvectors = calculate_eigen(cx)
    # unnormalized_eigenvectors= eigenvectors / eigenvectors[1:, :] # You can unnormalize it or just leave it alone :)
    return eigenvectors.T if unnormalized_eigenvectors is None else unnormalized_eigenvectors.T

def main():

    cx = np.array(
        [
            [7, -2],
            [-2, 11]
        ]
    )

    a_matrix = create_a_matrix(cx)
    print("eigen vectors (A matrix):\n", a_matrix, "\n" + 50 * "-")

    cy = a_matrix @ cx @ a_matrix.T
    print("cy Matrix:\n", cy, "\n" + 50 * "-")

    y1 = generate_random_floats_with_mean_variance(
        mean=0.0,
        variance=np.sqrt(cy[0][0])
    )

    y2 = generate_random_floats_with_mean_variance(
        mean=0.0,
        variance=np.sqrt(cy[1][1])
    )

    y_values_transposed = np.array(
        [
            y1,
            y2
        ]
    )
    cy_calculated_manually = np.cov(y_values_transposed)
    print("cy Matrix calculated manually:\n",cy_calculated_manually , "\n" + 50 * "-")

    new_x_values = np.linalg.inv(a_matrix) @ y_values_transposed
    print("cx Matrix calculated manually:\n",np.cov(new_x_values) , "\n" + 50 * "-")

if __name__ == "__main__":
    main()
