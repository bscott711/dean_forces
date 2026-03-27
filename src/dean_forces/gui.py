import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
try:
    from .app import DeanForcesSimulator, Geometry, DeanModel
except (ImportError, ValueError):
    from dean_forces.app import DeanForcesSimulator, Geometry, DeanModel

def run_gui():
    st.set_page_config(page_title="Dean Force Dashboard", layout="wide")

    st.title("Dean Microfluidic Force Simulator")
    st.markdown("""
    Explore the balance between **Inertial Lift ($F_L$)** and **Dean Drag ($F_D$)** in curved channels.
    $F_L/F_D > 1$ typically indicates a strong focusing regime.
    """)

    # Sidebar: Global Channel Geometry
    st.sidebar.header("Channel Geometry")
    width_um = st.sidebar.slider("Channel Width (μm)", 50.0, 500.0, 150.0, step=10.0)
    height_um = st.sidebar.slider("Channel Height (μm)", 50.0, 500.0, 150.0, step=10.0)
    
    # Initialize Simulator
    sim = DeanForcesSimulator(Geometry(width_um=width_um, height_um=height_um))

    # Sidebar: Model Config
    st.sidebar.header("Physics Model")
    model = st.sidebar.selectbox("Dean Model", [DeanModel.REZAI2017, DeanModel.SIMPLE])
    alpha = 0.30
    if model == DeanModel.SIMPLE:
        alpha = st.sidebar.slider("Simple alpha (calibration)", 0.05, 1.0, 0.30, step=0.05)

    # Main Tabs
    tab_p, tab_v, tab_s = st.tabs(["Particle Size Sweep", "Velocity Sweep", "Spiral Decay"])

    with tab_p:
        st.subheader("Varying Particle Size")
        col1, col2 = st.columns([1, 3])
        with col1:
            u_p = st.number_input("Fluid velocity (m/s)", 0.1, 5.0, 1.04, key="u_p")
            r_p = st.number_input("Radius R (mm)", 1.0, 50.0, 4.3, key="r_p")
            dp_range = st.slider("Particle diameter range (μm)", 1.0, 40.0, (1.0, 20.0))
        
        df_p = sim.particle_sweep(u_p, r_p, dp_range[0], dp_range[1], model, alpha, quiet=True)
        
        # Plotly
        fig_p = make_subplots(specs=[[{"secondary_y": True}]])
        fig_p.add_trace(go.Scatter(x=df_p["dp_um"], y=df_p["FL_pN"], name="Inertial Lift", line=dict(color="blue", width=3)))
        fig_p.add_trace(go.Scatter(x=df_p["dp_um"], y=df_p["FD_pN"], name="Dean Drag", line=dict(color="red", width=3)))
        fig_p.add_trace(go.Scatter(x=df_p["dp_um"], y=df_p["FL_over_FD"], name="Ratio (FL/FD)", line=dict(color="green", width=3, dash="dot")), secondary_y=True)
        
        fig_p.add_hline(y=1.0, line_dash="dash", line_color="black", secondary_y=True)
        fig_p.update_xaxes(title_text="Particle diameter (μm)", type="log")
        fig_p.update_yaxes(title_text="Force (pN)", type="log", secondary_y=False)
        fig_p.update_yaxes(title_text="Ratio", secondary_y=True)
        fig_p.update_layout(height=500, title="Force Balance vs Particle Size", hovermode="x unified")
        
        st.plotly_chart(fig_p, width="stretch")
        
        xc = sim._find_crossover(df_p["dp_um"].to_numpy(), df_p["FL_over_FD"].to_numpy())
        if xc:
            st.success(f"**Crossover diameter:** {xc:.2f} μm")
        else:
            st.warning("No crossover found in this range.")

    with tab_v:
        st.subheader("Varying Fluid Velocity")
        col1, col2 = st.columns([1, 3])
        with col1:
            dp_v = st.number_input("Particle size (μm)", 1.0, 50.0, 12.0, key="dp_v")
            r_v = st.number_input("Radius R (mm)", 1.0, 50.0, 4.3, key="r_v")
            u_range = st.slider("Velocity range (m/s)", 0.01, 5.0, (0.01, 1.2))
            
        df_v = sim.velocity_sweep(dp_v, r_v, u_range[0], u_range[1], model, alpha, quiet=True)
        
        fig_v = make_subplots(specs=[[{"secondary_y": True}]])
        fig_v.add_trace(go.Scatter(x=df_v["U_m_s"], y=df_v["FL_pN"], name="Lift", line=dict(color="blue", width=3)))
        fig_v.add_trace(go.Scatter(x=df_v["U_m_s"], y=df_v["FD_pN"], name="Drag", line=dict(color="red", width=3)))
        fig_v.add_trace(go.Scatter(x=df_v["U_m_s"], y=df_v["FL_over_FD"], name="Ratio", line=dict(color="green", width=3, dash="dot")), secondary_y=True)
        
        fig_v.add_hline(y=1.0, line_dash="dash", line_color="black", secondary_y=True)
        fig_v.update_xaxes(title_text="Mean Velocity (m/s)", type="log")
        fig_v.update_yaxes(title_text="Force (pN)", type="log")
        fig_v.update_layout(height=500, title="Velocity Sensitivity", hovermode="x unified")
        st.plotly_chart(fig_v, width="stretch")

    with tab_s:
        st.subheader("Spiral Decay (Varying Radius)")
        col1, col2 = st.columns([1, 3])
        with col1:
            u_s = st.number_input("Velocity (m/s)", 0.1, 5.0, 1.04, key="u_s")
            dp_s = st.number_input("Particle size (μm)", 1.0, 50.0, 12.0, key="dp_s")
            r_range = st.slider("Radius range (mm)", 1.0, 50.0, (4.3, 15.0))
            
        df_s = sim.spiral_sweep(u_s, dp_s, r_range[0], r_range[1], model, alpha, quiet=True)
        
        fig_s = make_subplots(specs=[[{"secondary_y": True}]])
        fig_s.add_trace(go.Scatter(x=df_s["R_mm"], y=df_s["FL_pN"], name="Lift", line=dict(color="blue", width=3)))
        fig_s.add_trace(go.Scatter(x=df_s["R_mm"], y=df_s["FD_pN"], name="Drag", line=dict(color="red", width=3)))
        fig_s.add_trace(go.Scatter(x=df_s["R_mm"], y=df_s["FL_over_FD"], name="Ratio", line=dict(color="green", width=3, dash="dot")), secondary_y=True)
        
        fig_s.update_xaxes(title_text="Radius of Curvature (mm)")
        fig_s.update_layout(height=500, title="Focusing Strength along Spiral Path", hovermode="x unified")
        st.plotly_chart(fig_s, width="stretch")
        
        gain = df_s["FL_over_FD"].iloc[-1] / df_s["FL_over_FD"].iloc[0]
        st.info(f"**Focusing Ratio Multiplier:** {gain:.2f}x gain as radius increases.")

    # Technical Details
    with st.expander("Show Physics Model Context"):
        template_path = Path(__file__).parent / "TEMPLATE.md"
        if template_path.exists():
            text = template_path.read_text()
            # Extract only the "Physics Model" content
            if "## Physics Model" in text:
                # Keep from ## Physics Model until the first separator
                physics_content = text.split("## Physics Model")[-1].split("---")[0]
                st.markdown("## Physics Model" + physics_content)
            else:
                st.markdown(text)

if __name__ == "__main__":
    run_gui()
