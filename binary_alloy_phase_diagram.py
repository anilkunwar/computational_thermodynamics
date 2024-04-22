import streamlit as st
import matplotlib.pyplot as plt
from pycalphad import Database, binplot
import pycalphad.variables as v
import tempfile
import os

def phase_diagram_app():
    st.title('Binary Alloy Phase Diagram Construction')
    st.write('This app constructs the alloy phase diagram when the names of the two elements are specified, and the TDB file is uploaded.')
    # User input for binary system elements
    element1 = st.text_input("Enter the name of the first element:")
    element2 = st.text_input("Enter the name of the second element:")
    
    uploaded_file = st.file_uploader("Upload TDB file", type=["tdb", "TDB"])
    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.getvalue())
        temp_file.close()
        
        # Load the database from the temporary file
        db = Database(temp_file.name)
        
        # Get available phases from the database
        available_phases = db.phases.keys()
        
        # Allow user to choose phases
        selected_phases = st.multiselect("Select phases:", list(available_phases))
        
        if selected_phases:
            # Temperature range
            min_temp = st.slider('Minimum Temperature (K)', 300, 800, 300)
            max_temp = st.slider('Maximum Temperature (K)', 300, 2000, 1600)
            
            # Composition range
            min_comp = st.slider(f'Minimum Composition (X_{element2.upper()})', 0.0, 1.0, 0.0, step=0.05)
            max_comp = st.slider(f'Maximum Composition (X_{element2.upper()})', 0.0, 1.0, 1.0, step=0.05)
            
            # Plot the phase diagram
            fig = plt.figure(figsize=(9, 6))
            axes = fig.gca()
            binplot(db, [element1.upper(), element2.upper(), 'VA'], selected_phases, {v.X(element2.upper()): (min_comp, max_comp, 0.05), v.T: (min_temp, max_temp, 10), v.P: 101325, v.N: 1}, plot_kwargs={'ax': axes})
            st.pyplot(fig)
        
        # Remove the temporary file after use
        os.unlink(temp_file.name)

if __name__ == '__main__':
    phase_diagram_app()

