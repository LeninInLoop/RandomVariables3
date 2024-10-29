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


def calculate_cov_manually(array: np.ndarray) -> np.ndarray:
    """
    Calculate 2x2 covariance matrix manually.

    Args:
        array: Input array of shape (2, n_samples), where 2 represents the two features
               and n_samples represents the number of observations.

    Returns:
        2x2 Covariance matrix
    """
    # Step 1: Get the number of samples (observations)
    # array.shape[1] gives us the number of columns, which corresponds to the number of samples.
    # Since our input array is of shape (2, n_samples), the second dimension is the number of samples.
    n_samples = array.shape[1]

    # Step 2: Center the data
    # To center the data, we subtract the mean of each feature from the corresponding feature values.
    # mean(axis=1) calculates the mean for each row (i.e., each feature).
    means = array.mean(axis=1)

    # We reshape the means to have shape (2, 1), so they can be broadcasted appropriately when
    # subtracting from the original array. Broadcasting allows element-wise subtraction across
    # each feature.
    centered_data = array - means.reshape(-1, 1)

    # Step 3: Initialize an empty 2x2 covariance matrix with zeros
    cov_matrix = np.zeros((2, 2))

    # Step 4: Calculate the variances (diagonal elements of the covariance matrix)
    # Variance is the average of the squared differences from the mean.
    # We calculate variance for the first feature (row 0) and the second feature (row 1).

    # Variance of the first feature (row 0)
    cov_matrix[0, 0] = np.sum(centered_data[0] * centered_data[0]) / (n_samples - 1)

    # Variance of the second feature (row 1)
    cov_matrix[1, 1] = np.sum(centered_data[1] * centered_data[1]) / (n_samples - 1)

    # Step 5: Calculate the covariance (off-diagonal elements of the covariance matrix)
    # Covariance measures how much two random variables vary together. It is calculated as
    # the average product of the deviations of each pair of corresponding values from their means.

    # Covariance between feature 1 and feature 2
    covariance = np.sum(centered_data[0] * centered_data[1]) / (n_samples - 1)

    # The covariance matrix is symmetric, so we assign the same calculated covariance value
    # to both off-diagonal elements.
    cov_matrix[0, 1] = covariance
    cov_matrix[1, 0] = covariance

    return cov_matrix

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
    print("cy Matrix calculated from random variables inside the vector:\n",cy_calculated_manually , "\n" + 50 * "-")

    new_x_values = np.linalg.inv(a_matrix) @ y_values_transposed
    print("cx Matrix calculated from random variables inside the vector:\n",np.cov(new_x_values) , "\n" + 50 * "-")

    print("cx Matrix calculated manually:\n",calculate_cov_manually(new_x_values) , "\n" + 50 * "-")

if __name__ == "__main__":
    main()
