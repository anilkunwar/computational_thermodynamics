import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from io import StringIO

# Function to generate free energy data
def generate_freeenergy_data(mole_fractions_peak, y_peak, A, num_points=100, component_names=None):
    num_components = len(mole_fractions_peak)
    
    if component_names is None:
        component_names = [f'x_E{i+1}' for i in range(num_components)]
    
    # Create linspace for each component
    compositions = {comp: np.linspace(0, 1, num_points) for comp in component_names}
    
    # Create meshgrid for multi-component system
    mesh = np.meshgrid(*[compositions[comp] for comp in component_names])
    
    # Compute Gibbs energy using the parabola function
    C = y_peak  # Set C as the peak value
    gibbs_energy = sum([A * (mesh[i] - mole_fractions_peak[i])**2 for i in range(num_components)]) + C
    
    # Flatten the data for DataFrame creation
    data = pd.DataFrame({component_names[i]: mesh[i].flatten() for i in range(num_components)})
    data['Gibbs Energy (J/mol)'] = gibbs_energy.flatten()
    
    return data

# Function to plot free energy for different component systems
def plot_free_energy(data, component_names, color_range):
    num_components = len(component_names)

    if num_components == 1:
        fig = go.Figure(go.Scatter(x=data[component_names[0]], y=data['Gibbs Energy (J/mol)'], mode='lines'))
        fig.update_layout(xaxis_title=f'{component_names[0]} Mole Fraction', yaxis_title='Gibbs Energy (J/mol)')
        st.plotly_chart(fig)

    elif num_components == 2:
        fig = go.Figure(go.Scatter3d(x=data[component_names[0]], y=data[component_names[1]], z=data['Gibbs Energy (J/mol)'],
                                     mode='markers', marker=dict(size=5, color=data['Gibbs Energy (J/mol)'], 
                                     colorscale=color_range)))
        fig.update_layout(scene=dict(
            xaxis_title=f'{component_names[0]} Mole Fraction',
            yaxis_title=f'{component_names[1]} Mole Fraction',
            zaxis_title='Gibbs Energy (J/mol)'
        ))
        st.plotly_chart(fig)

    elif num_components == 3:
        fig = go.Figure(go.Scatter3d(x=data[component_names[0]], y=data[component_names[1]], z=data['Gibbs Energy (J/mol)'],
                                     mode='markers', marker=dict(size=5, color=data[component_names[2]], 
                                     colorscale=color_range)))
        fig.update_layout(scene=dict(
            xaxis_title=f'{component_names[0]} Mole Fraction',
            yaxis_title=f'{component_names[1]} Mole Fraction',
            zaxis_title=f'{component_names[2]} Mole Fraction',
        ))
        st.plotly_chart(fig)

    elif num_components == 4:
        fig = go.Figure(go.Scatter3d(x=data[component_names[0]], y=data[component_names[1]], z=data[component_names[2]],
                                     mode='markers', marker=dict(size=5, color=data['Gibbs Energy (J/mol)'], 
                                     colorscale=color_range)))
        fig.update_layout(scene=dict(
            xaxis_title=f'{component_names[0]} Mole Fraction',
            yaxis_title=f'{component_names[1]} Mole Fraction',
            zaxis_title=f'{component_names[2]} Mole Fraction',
        ))
        st.plotly_chart(fig)

    else:
        raise ValueError(f'Unable to visualize a system with {num_components} components.')

# Function to download data as CSV
def download_csv(data):
    csv_buffer = StringIO()
    data.to_csv(csv_buffer, index=False)
    st.download_button(label="Download data as CSV", data=csv_buffer.getvalue(), file_name="free_energy_data.csv", mime="text/csv")

# Streamlit app
st.title("Gibbs Free Energy Surface Reconstruction")

# Sidebar inputs
st.sidebar.header("Input Parameters")
A = st.sidebar.slider("Parabola Coefficient (A)", min_value=-1.0E+10, max_value=-1.0E+01, value=-1.0E+06, step=1.0E+05, format="%.2E")
y_peak = st.sidebar.number_input("Peak Gibbs Energy (J/mol)", value=-2.513796E+03, step=1.0E+02,  format="%.2E")
num_points = st.sidebar.slider("Number of Points", min_value=10, max_value=100, value=50, step=5)

# Mole fractions peak input
num_components = st.sidebar.slider("Number of Components", min_value=1, max_value=4, value=2)
mole_fractions_peak = []
component_names = []

for i in range(num_components):
    mole_fractions_peak.append(st.sidebar.slider(f"Mole Fraction Peak for Component {i+1}", 0.0, 1.0, 0.5))
    component_names.append(f"x_E{i+1}")

# Color range selection
st.sidebar.header("Color Options")
min_color = st.sidebar.color_picker("Pick Minimum Color", "#0000ff")  # Blue
max_color = st.sidebar.color_picker("Pick Maximum Color", "#ff0000")  # Red
color_range = [[0, min_color], [1, max_color]]

# Generate data
data = generate_freeenergy_data(mole_fractions_peak, y_peak, A, num_points, component_names=component_names)

# Plot data
plot_free_energy(data, component_names, color_range)

# Download CSV button
download_csv(data)

