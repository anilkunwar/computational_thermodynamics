import streamlit as st
from pycalphad import Database, calculate, variables as v
from pycalphad.plot.utils import phase_legend
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from io import StringIO
import base64
#from parabola_module import generate_parabola_data  # Importing the parabola generation module
from data_reconstruction import generate_freeenergy_data  # Importing the parabola generation module

# Define the parabola function for fitting
def parabola(X, *params):
    N = len(X) - 1
    x0eq = params[:N]
    coeffs = params[N:]
    
    f = np.zeros(X[0].shape)
    for i in range(N):
        f += coeffs[i] * (X[i] - x0eq[i])**2
    return f

# Set up the Streamlit app
st.title('TDB File Analyzer for Composition Dependent Gibbs Free Energy Computation')

# Step 1: Upload .TDB file
uploaded_file = st.file_uploader("Upload your .TDB file", type=['TDB', 'tdb'])

if uploaded_file is not None:
    # Step 2: Read the uploaded TDB file
    with open('uploaded_database.tdb', 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    st.success('File uploaded successfully!')
    
    # Load the database
    dbf = Database('uploaded_database.tdb')
    
    # Step 3: Display the list of available phases
    phases = list(dbf.phases.keys())
    st.write("Available phases in the database:")
    selected_phases = st.multiselect('Select phases to calculate', phases)
    
    # Step 4: Display the list of available components
    all_comps = list(dbf.elements)
    st.write("Available components in the database:")
    selected_comps = st.multiselect('Select components to include', all_comps)
    
    if selected_phases and selected_comps:
        # Ensure 'VA' is not included in selected components
        if 'VA' in selected_comps:
            selected_comps.remove('VA')

        # Request user to input temperature
        temperature = st.number_input('Input temperature (K)', value=298, format="%.2f")
        # Request user to input composition ranges for filtering
        comp_ranges = {}
        for comp in sorted(selected_comps):
            L = st.number_input(f'Lower bound for {comp}', value=0.0, format="%.4f")
            H = st.number_input(f'Upper bound for {comp}', value=1.0, format="%.4f")
            comp_ranges[comp] = (L, H)
        
        for phase in selected_phases:
            calc_result = calculate(dbf, selected_comps + ['VA'], phase, P=101325, T=[temperature])
            st.write(f"Calculation result for phase: {phase}")
            
            # Get the composition in mole fraction
            xcomp = calc_result.X.values
            st.write('shape of the composition matrix',xcomp.shape)
            # Get the gibbs energy 
            gibbs = calc_result.GM.values
            st.write('shape of the gibbs matrix',gibbs.shape)
            
            # Handle reshaping based on the shape of xcomp
            shape = xcomp.shape
            if len(shape) == 5:
                # Reshape the composition array
                num_points = shape[3]
                num_comps = shape[4]
                xcompr = xcomp.reshape((num_points, num_comps))
                st.write('shape of the reshaped  composition matrix',xcompr.shape)
                
                # Create DataFrame with component names in alphabetical order
                sorted_comps = sorted(selected_comps)
                column_names = [f'x{comp}' for comp in sorted_comps]
                dfx_comp = pd.DataFrame(xcompr, columns=column_names)
                
                # Reshape the Gibbs free energy array
                gibbsr = gibbs.reshape(num_points)
                st.write('shape of the reshaped gibbs matrix',gibbsr.shape)

                # Create DataFrame for Gibbs free energy
                dfgibbs = pd.DataFrame(gibbsr, columns=[f'gm{phase}'])
                
                # Merge the composition DataFrame with the Gibbs free energy DataFrame
                df1 = dfx_comp.copy()
                df1[f'gm{phase}'] = dfgibbs[f'gm{phase}']
                
                # Apply composition range filters
                df_filtered = df1
                for comp, (L, H) in comp_ranges.items():
                    df_filtered = df_filtered.loc[(df_filtered[f'x{comp}'] >= L) & (df_filtered[f'x{comp}'] <= H)]
                
                # Display the filtered DataFrame
                st.write(df_filtered.head())
                st.write('shape of gibbs matrix for selected composition',df_filtered.shape)
                # If there is only one data point for Gibbs free energy, generate parabola data
                if gibbsr.shape == (1,):
                    st.warning("Only one Gibbs free energy point available. Generating parabolic data.")
                    
                    # Request user input for parabola generation
                    peak_x = st.number_input("Enter the mole fraction for the peak (c):", min_value=0.0, max_value=1.0, value=0.5)
                    peak_y = st.number_input("Enter the Gibbs energy for the peak (J/mol):", value=-2.5E+04, format="%.2E")
                    A = st.number_input("Enter the coefficient A for the parabola (J/mol):", value=1.0E+06, format="%.2E")
                    num_points_parabola = st.slider("Number of data points for parabola:", min_value=10, max_value=200, value=100)

                    # Generate parabola data
                    parabola_data = generate_parabola_data(peak_x, peak_y, A, num_points_parabola)

                    # Display the generated parabola data
                    st.write(parabola_data.head())

                    # Provide a download link for the generated data
                    csv_parabola = parabola_data.to_csv(index=False)
                    b64_parabola = base64.b64encode(csv_parabola.encode()).decode()
                    href_parabola = f'<a href="data:file/csv;base64,{b64_parabola}" download="reconstructed_data.csv">Download Parabolic Data</a>'
                    st.markdown(href_parabola, unsafe_allow_html=True)
                else:
                    # Add download button for CSV file
                    csv = df_filtered.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
                    href = f'<a href="data:file/csv;base64,{b64}" download="direct_data.csv">Download CSV File</a>'
                    st.markdown(href, unsafe_allow_html=True)
                
                # Get fitting parameters from user
                x0eq = []
                for comp in sorted_comps:    
                    x0 = st.number_input(f'Value for x0eq_{comp}', value=0.0, format="%.4f")    
                    x0eq.append(x0)    
                    
                num_coeffs = len(sorted_comps)
                guess = [st.number_input(f'Coefficient {i+1}', value=0.0, format="%.4f") for i in range(num_coeffs)]    
                    
                # Prepare data for curve fitting
                X = [df_filtered[f'x{comp}'].values for comp in sorted_comps]
                Y = df_filtered[f'gm{phase}'].values
                    
                try:
                    # Perform curve fitting
                    popt, pcov = curve_fit(parabola, X, Y, p0=guess + x0eq)
                    st.write("Fitted Parameters:", popt)
                    st.write("Covariance Matrix:", pcov)
                except Exception as e:
                    st.write("Error in curve fitting:", e)
            else:
                st.write("Unexpected shape for xcomp:", xcomp.shape)
                st.write("Unexpected shape for gibbs:", gibbs.shape)

