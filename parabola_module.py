import numpy as np
import pandas as pd

def generate_parabola_data(x_peak, y_peak, A, num_points=100):
    """
    Generates the parabola data based on the given parameters.
    
    Args:
        x_peak (float): The mole fraction at the peak of the parabola.
        y_peak (float): The Gibbs energy at the peak (in J/mol).
        A (float): The coefficient that controls the width of the parabola.
        num_points (int): The number of data points to generate.
    
    Returns:
        pd.DataFrame: A DataFrame containing the composition (c) and Gibbs energy data.
    """
    # Set C as the peak value
    C = y_peak
    # Generate data points for the parabola
    x = np.linspace(0, 1, num_points)
    y = A * (x - x_peak) ** 2 + C
    
    # Create a DataFrame with the data
    data = pd.DataFrame({
        'Composition (c)': x,
        'Gibbs Energy (J/mol)': y
    })
    
    return data

