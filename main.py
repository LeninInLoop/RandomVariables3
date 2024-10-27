# Standard Library Imports
from collections import Counter

# Type Hinting Imports
from typing import Dict, List, Tuple, Union, Any

# Third-Party Library Imports
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

# Counting methods
METHOD_MANUAL = 0
METHOD_COUNTER_CLASS = 1

def round_to_decimals(value: float, decimal_places: int = 0) -> float:
    """
    Custom round function to round a number to the specified number of decimal places.

    This function handles both positive and negative numbers correctly, ensuring
    that numbers are rounded to the nearest value, providing consistent results for all inputs.

    Parameters:
    value (float): The number to be rounded.
    decimal_places (int): The number of decimal places to round to (default is 0).

    Returns:
    float: The rounded number.
    """
    multiplier = 10 ** decimal_places
    if value >= 0:
        return (value * multiplier + 0.5) // 1 / multiplier
    else:
        return -((-value * multiplier + 0.5) // 1) / multiplier

def display_results(value_counts: Dict[float, int]) -> None:
    """
    Display the frequency and percentage of values in a user-friendly format.

    Parameters:
    value_counts (Dict[float, int]): A dictionary where keys are value bins and values are frequencies.
    total_count (int): The total number of samples (default is 1,000,000).

    The function prints:
    - Value ranges (with special handling for 0 and 1).
    - The frequency of occurrences for each bin.
    - The percentage of each bin's occurrences relative to the total count.
    """
    print(" " * 9 + f"{'Value':<13} {'Frequency':<12} {'Percentage':<10}")
    print("-" * 50)

    total_count = 10**6
    for value, count in sorted(value_counts.items()):
        percentage = (count / total_count) * 100
        if value == 0:
            lower_bound = value
            upper_bound = round_to_decimals(value + 0.0005, 4)
            print(f"{lower_bound:0.4f} <= x < {upper_bound:<10} {count:<12} {percentage:<10.2f}")
        elif value == 1:
            lower_bound = round_to_decimals(value - 0.0005, 4)
            upper_bound = value
            print(f"{lower_bound} <= x <= {upper_bound:<9} {count:<12} {percentage:<10.2f}")
        else:
            lower_bound = round_to_decimals(value - 0.0005, 4)
            upper_bound = round_to_decimals(value + 0.0005, 4)
            print(f"{lower_bound} <= x < {upper_bound:<10} {count:<12} {percentage:<10.2f}")


def generate_random_floats_with_mean_variance(mean: float = 0.0, variance: float = 0.0) -> List[float]:
    return ((np.array(variance) * np.random.randn( 10 ** 6 )) + np.array(mean)) .tolist()


def initialize_value_count_dict(x_range:Tuple[Union[int],Union[int]] = (0,1)) -> Dict[float, int]:
    """
    Initialize a dictionary for counting occurrences of each bin value.

    Returns:
    Dict[float, int]: A dictionary with bin values as keys and 0 as initial counts.
    """
    values = np.linspace(x_range[0],x_range[1],1000).tolist()
    return {round_to_decimals(value, 3):0 for value in values}

def count_values_by_method(
        random_floats: List[float],
        method: int,
        x_range:Tuple[Union[int],Union[int]] = (0,1)
) -> Dict[float, int]:
    """
    Count occurrences of rounded random floats in the specified value bins using a specified method.

    Parameters:
    value_bins (List[float]): A list of bin values to count occurrences against.
    random_floats (List[float]): A list of random float values to be counted.
    method (int): The counting method to use.
                  METHOD_MANUAL (1) uses a manual O(n^2) algorithm.
                  METHOD_COUNTER_CLASS (2) uses Python's Counter class for an O(n log n) approach.

    Returns:
    Dict[float, int]: A dictionary where keys are bin values and values are the counts of occurrences.
    """
    value_count_dict = initialize_value_count_dict(x_range)
    rounded_random_floats = [round_to_decimals(value, 3) for value in random_floats]
    if method == METHOD_MANUAL:  # O(n^2) Algorithm
        for bin_value in value_count_dict.keys():
            for random_float in rounded_random_floats:
                if random_float == bin_value:
                    value_count_dict[bin_value] += 1
    elif method == METHOD_COUNTER_CLASS:  # O(n log n) Algorithm
        random_counts = Counter(rounded_random_floats)
        for number, count in random_counts.items():
            if number in value_count_dict:
                value_count_dict[number] += count
    return value_count_dict

def calculate_cdf_from_pdf(pdf_values: List[float], x_values: List[float]) -> np.ndarray:
    """
    Calculates the cumulative distribution function (CDF) from the probability density function (PDF).

    Parameters:
    pdf_values (List[float]): The values representing the probability density function (PDF).
    x_values (List[float]): The corresponding x values.

    Returns:
    List[float]: The calculated CDF values.
    """
    # Use cumulative trapezoidal integration to calculate the CDF
    return integrate.cumulative_trapezoid(pdf_values, x_values, dx=0.001, initial=0)


def save_pdf_and_cdf_plot_from_pdf(
        value_counts: Dict[float, int],
        display: bool,
        file_name: str = "pdf_and_cdf_plot.png",
):
    """
    Saves a plot with both the Probability Density Function (PDF) in two formats:
    a histogram-based PDF and a line plot, as well as the Cumulative Distribution Function (CDF).

    Parameters:
    value_counts (Dict[float, int]): A dictionary where keys are x-values and values are frequencies (PDF).
    display (bool): If True, display the plot interactively; if False, save the plot as an image.
    filename (str): The name of the file to save the plot as (default is 'pdf_and_cdf_plot.png').
    show_histogram (bool, optional): If True, include a histogram representation of the PDF. Default is True.
    x_range (Tuple[int, int], optional): The range of x-values to use. If None, uses the keys from value_counts. Default is None.

    The function performs the following:
    - If show_histogram is True, plots a histogram to represent the PDF (using the frequency from value_counts).
    - Plots the PDF as a smooth line plot.
    - Computes and plots the CDF based on the PDF data.
    - Adjusts the plot layout based on whether the histogram is shown or not.
    - If x_range is provided, uses it to create value bins; otherwise, uses the keys from value_counts.
    - Displays the plot if `display` is True, or saves it to a file if `display` is False.
    """

    x_values = list(value_counts.keys())
    y_values = [count / 1000 for count in value_counts.values()]

    plt.figure(figsize=(12, 10))

    fig, (ax_pdf, ax_cdf) = plt.subplots(2, 1, figsize=(12, 12), height_ratios=[1, 1])
    fig.subplots_adjust(hspace=0.4)  # Increase space between subplots

    # Plot PDF without Histogram
    ax_pdf.plot(x_values[1:-1], y_values[1:-1], color='blue')
    ax_pdf.fill_between(x_values[1:-1], y_values[1:-1], alpha=0.3)
    ax_pdf.set_xlabel('X Values\n\n')
    ax_pdf.set_ylabel('PDF')
    ax_pdf.set_title('PDF: Probability Density Function')

    # Calculate CDF
    cdf_values = calculate_cdf_from_pdf(y_values, x_values)

    ax_cdf.plot(x_values, cdf_values, color='red')
    ax_cdf.set_xlabel('X Values')
    ax_cdf.set_ylabel('CDF')
    ax_cdf.set_title('CDF: Cumulative Distribution Function')

    # Add overall title
    fig.suptitle('Probability Density Function and Cumulative Distribution Function', fontsize=16)

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(file_name)
    if display:
        plt.show()
    else:
        plt.close()


def calculate_eigen(matrix: np.ndarray) -> tuple[Any, Any]:
    # Check if matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square")
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    return eigenvalues, eigenvectors


def create_a_matrix(cx: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = calculate_eigen(cx)
    unnormalized_eigenvectors= eigenvectors / eigenvectors[1:, :]
    return unnormalized_eigenvectors.T

def main():

    # Generate X and Y values with normal pdf and zero mean and variance equal to one.
    random_floats_with_mean_zero_variance1 = generate_random_floats_with_mean_variance(
        mean=0.0,
        variance=np.sqrt(1)
    )
    random_floats_with_mean_zero_variance2 = generate_random_floats_with_mean_variance(
        mean=0.0,
        variance=np.sqrt(1)
    )

    # Count the values then plot the X values to show that randn values are normal with zero mean and variance of one.
    counted_values = count_values_by_method(
        random_floats_with_mean_zero_variance1,
        method=METHOD_COUNTER_CLASS,
        x_range=(-6, 6)
    )
    display_results(counted_values)

    save_pdf_and_cdf_plot_from_pdf(
        counted_values,
        display=True,
        file_name="randn_pdf_and_cdf_plot.png"
    )

    cx = np.array(
        [
            [7, -2],
            [-2, 11]
        ]
    )

    a_matrix = create_a_matrix(cx)
    print(a_matrix)

    cy = a_matrix @ cx @ a_matrix.T
    print(cy)

    random_floats_with_mean_with_variance = generate_random_floats_with_mean_variance(
        mean=0.0,
        variance=np.sqrt(cy[0][0])
    )

    random_floats_with_mean_with_variance1 = generate_random_floats_with_mean_variance(
        mean=0.0,
        variance=np.sqrt(cy[1][1])
    )

    y_values_transposed = np.array(
        [
            random_floats_with_mean_with_variance,
            random_floats_with_mean_with_variance1
        ]
    )

    new_x_values = np.linalg.inv(a_matrix) @ y_values_transposed
    print(np.cov(new_x_values))

if __name__ == "__main__":
    main()
