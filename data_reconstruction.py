import numpy as np
import pandas as pd

def generate_freeenergy_data(mole_fractions_peak, y_peak, A, num_points=100, component_names=None):
    """
    Generates the free energy data based on the given parameters for multiple components.
    
    Args:
        mole_fractions_peak (list of floats): The mole fractions at the peak of the parabola for each component.
        y_peak (float): The Gibbs energy at the peak (in J/mol).
        A (float): The coefficient that controls the width of the parabola.
        num_points (int): The number of data points to generate.
        component_names (list of str): The names of the components.
    
    Returns:
        pd.DataFrame: A DataFrame containing the compositions and Gibbs energy data for multiple components.
    """
    num_components = len(mole_fractions_peak)
    
    if component_names is None:
        component_names = [f'element{i+1}' for i in range(num_components)]
    
    # Create a set of points for each component
    compositions = {comp: np.linspace(0, 1, num_points) for comp in component_names}
    
    # Compute the parabola values for each composition
    C = y_peak  # Set C as the peak value
    gibbs_energy = sum([A * (compositions[comp] - mole_fractions_peak[i])**2 for i, comp in enumerate(component_names)]) + C
    
    # Prepare the data into a DataFrame
    data = pd.DataFrame(compositions)
    data['Gibbs Energy (J/mol)'] = gibbs_energy
    
    return data
