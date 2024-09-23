import numpy as np
import streamlit as st
import plotly.graph_objs as go

# Function to compute free energy
def free_energy(x, y, A_Fe, A_Cr, B_Fe, B_Cr):
    G = (A_Fe * x**2 * (0.94 - x)**2) + \
        (A_Cr * y**2 * (0.05 - y)**2) + \
        B_Fe * (x**2 + 6 * (0.94 - x) - 4 * (1.94 - x)) + \
        B_Cr * (y**2 + 6 * (0.05 - y) - 4 * (1.94 - y))
    return G

# Sidebar for sliders to adjust constants
st.sidebar.header("Adjust the Constants")
A_Fe = st.sidebar.slider("A_Fe", min_value=-100.0, max_value=100.0, value=1.0, step=0.1)
A_Cr = st.sidebar.slider("A_Cr", min_value=-100.0, max_value=100.0, value=1.0, step=0.1)
B_Fe = st.sidebar.slider("B_Fe", min_value=-20.0, max_value=20.0, value=1.0, step=0.1)
B_Cr = st.sidebar.slider("B_Cr", min_value=-20.0, max_value=20.0, value=1.0, step=0.1)

# Sliders for adjusting font sizes
st.sidebar.header("Adjust Font Sizes")
axis_title_font_size = st.sidebar.slider("Axis Title Font Size", min_value=10, max_value=30, value=16, step=1)
tick_label_font_size = st.sidebar.slider("Tick Label Font Size", min_value=8, max_value=20, value=12, step=1)
colorbar_title_font_size = st.sidebar.slider("Colorbar Title Font Size", min_value=10, max_value=30, value=16, step=1)
colorbar_tick_label_font_size = st.sidebar.slider("Colorbar Tick Label Font Size", min_value=8, max_value=20, value=12, step=1)

# Text input fields for custom axis labels
st.sidebar.header("Custom Axis and Color Scale Bar Labels")
x_label = st.sidebar.text_input("X-axis Label", "c_Fe (x)")
y_label = st.sidebar.text_input("Y-axis Label", "c_Cr (y)")
z_label = st.sidebar.text_input("Z-axis Label", "G (Free Energy)")
colorscale_label = st.sidebar.text_input("Color Scale Bar Label", "Free Energy")

# Color scale picker
st.sidebar.header("Choose a Color Scale")
color_scales = ['Jet', 'Viridis', 'Cividis', 'Inferno', 'Plasma', 'Blues', 'Electric']
color_scale = st.sidebar.selectbox("Color Scale", color_scales)

# Display LaTeX equation with the numerical coefficients
st.latex(rf"""
G(x, y) = {A_Fe} \cdot x^2 (0.94 - x)^2 + {A_Cr} \cdot y^2 (0.05 - y)^2 + 
{B_Fe} \left(x^2 + 6(0.94 - x) - 4(1.94 - x)\right) + 
{B_Cr} \left(y^2 + 6(0.05 - y) - 4(1.94 - y)\right)
""")

# Meshgrid for x and y values
x = np.linspace(0, 1, 50)
y = np.linspace(0, 1, 50)
X, Y = np.meshgrid(x, y)

# Calculate free energy
G = free_energy(X, Y, A_Fe, A_Cr, B_Fe, B_Cr)

# Create 3D plot using Plotly
#fig = go.Figure(data=[go.Surface(z=G, x=X, y=Y, colorscale=color_scale, colorbar=dict(
#    title="Free Energy",  # Colorbar title
#    titlefont=dict(size=colorbar_title_font_size),  # Colorbar title font size
#    tickfont=dict(size=colorbar_tick_label_font_size)  # Colorbar tick label font size
#))])
fig = go.Figure(data=[go.Surface(z=G, x=X, y=Y, colorscale=color_scale, colorbar=dict(
    title=colorscale_label,  # Colorbar title
    titlefont=dict(size=colorbar_title_font_size),  # Colorbar title font size
    tickfont=dict(size=colorbar_tick_label_font_size)  # Colorbar tick label font size
))])


# Customize plot with adjustable font sizes and custom axis labels
fig.update_layout(
    title="3D Free Energy Surface",
    scene=dict(
        xaxis_title=x_label,
        yaxis_title=y_label,
        zaxis_title=z_label,
        xaxis=dict(title_font=dict(size=axis_title_font_size), tickfont=dict(size=tick_label_font_size)),  # X-axis
        yaxis=dict(title_font=dict(size=axis_title_font_size), tickfont=dict(size=tick_label_font_size)),  # Y-axis
        zaxis=dict(title_font=dict(size=axis_title_font_size), tickfont=dict(size=tick_label_font_size)),  # Z-axis
    ),
    width=700,
    height=700,
)

# Display plot in Streamlit
st.plotly_chart(fig)

