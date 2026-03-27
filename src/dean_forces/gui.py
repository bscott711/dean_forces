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
    alpha = 0.30  # Default alpha
    if model == DeanModel.SIMPLE:
        alpha = st.sidebar.slider("Simple alpha (calibration)", 0.05, 1.0, 0.30, step=0.05)
    
    selected_cmap = st.sidebar.selectbox("Colormap Theme", ["viridis", "plasma", "magma", "inferno", "cividis", "turbo"], index=0)

    # Main Tabs - REORDERED: Design Heatmap first
    tab_h, tab_p, tab_v, tab_s = st.tabs(["Design Heatmap", "Particle Size Sweep", "Velocity Sweep", "Spiral Decay"])

    # --- TAB H: Design Heatmap ---
    with tab_h:
        st.header("Optimization Sweep (Width vs Velocity)")
        st.markdown("""
        Find the **optimal operating window** by sweeping channel dimensions and flow rates.
        This heatmap uses a composite score balancing **Dean Transport**, **Outlet Focusing**, and **Robustness**.
        """)
        
        with st.expander("Sweep Parameters", expanded=True):
            dp_h = st.number_input("Target cell diameter (μm)", 1.0, 50.0, 12.0, key="dp_h")
            w_range = st.slider("Width range (μm)", 50.0, 500.0, (75.0, 250.0))
            v_range = st.slider("Velocity range (m/s)", 0.01, 5.0, (0.1, 1.5))
            r_range_h = st.slider("Spiral radius range (mm)", 1.0, 30.0, (2.0, 15.0))
            res = st.select_slider("Grid resolution", options=[10, 25, 50, 75, 100], value=100)
            run_sweep = st.button("Run Design Sweep", type="primary")

        if run_sweep:
            with st.spinner(f"Simulating {res*res} configurations..."):
                rdf = sim.design_sweep(
                    dp_um=dp_h,
                    height_um=height_um,
                    width_start_um=w_range[0],
                    width_end_um=w_range[1],
                    n_width=res,
                    u_start=v_range[0],
                    u_end=v_range[1],
                    n_u=res,
                    r_start_mm=r_range_h[0],
                    r_end_mm=r_range_h[1],
                    model=model,
                    alpha=alpha,
                )
                st.session_state["design_sweep_results"] = rdf
        
        if "design_sweep_results" in st.session_state:
            rdf = st.session_state["design_sweep_results"]
            
            # Display Heatmaps
            def make_hm(df, z_col, title, row, col, show_scale=True, cb_x=1.02, cb_y=0.5, cb_len=0.45):
                pivot = df.pivot(index="U_m_s", columns="width_um", values=z_col)
                return go.Heatmap(
                    z=pivot.values,
                    x=pivot.columns,
                    y=pivot.index,
                    colorscale=selected_cmap,
                    showscale=show_scale,
                    colorbar=dict(
                        title=dict(text=title, side="right"), 
                        thickness=15,
                        x=cb_x,
                        y=cb_y,
                        len=cb_len
                    ),
                    hovertemplate="Width: %{x}μm<br>U: %{y}m/s<br>Value: %{z:.3f}<extra></extra>"
                )

            view_mode = st.selectbox("Zoom into specific metric:", ["All (2x2 Grid)", "Composite Score", "Mean Ud", "Outlet Ratio", "Max De"], key="view_mode")
            
            if view_mode == "All (2x2 Grid)":
                fig_h = make_subplots(rows=2, cols=2, subplot_titles=[
                    "Composite Design Score", "Mean Dean Velocity (mm/s)", 
                    "Outlet FL/FD", "Max Dean Number"
                ], horizontal_spacing=0.15, vertical_spacing=0.15)
                
                fig_h.add_trace(make_hm(rdf, "score", "Score", 1, 1, cb_x=1.02, cb_y=0.8, cb_len=0.4), row=1, col=1)
                fig_h.add_trace(make_hm(rdf, "mean_Ud_mm_s", "Ud", 1, 2, cb_x=1.12, cb_y=0.8, cb_len=0.4), row=1, col=2)
                fig_h.add_trace(make_hm(rdf, "outlet_FL_over_FD", "Ratio", 2, 1, cb_x=1.02, cb_y=0.28, cb_len=0.4), row=2, col=1)
                fig_h.add_trace(make_hm(rdf, "max_De", "De", 2, 2, cb_x=1.12, cb_y=0.28, cb_len=0.4), row=2, col=2)
                
                fig_h.update_xaxes(matches='x')
                fig_h.update_yaxes(matches='y')
                fig_h.update_layout(height=800, margin=dict(t=50, b=50, l=50, r=150))
            else:
                metric_map = {
                    "Composite Score": ("score", "Score"),
                    "Mean Ud": ("mean_Ud_mm_s", "Ud (mm/s)"),
                    "Outlet Ratio": ("outlet_FL_over_FD", "Outlet FL/FD"),
                    "Max De": ("max_De", "Max Dean Number")
                }
                z_key, z_title = metric_map[view_mode]
                fig_h = go.Figure(data=[make_hm(rdf, z_key, z_title, 1, 1, cb_x=1.05, cb_y=0.5, cb_len=0.9)])
                fig_h.update_layout(height=600, title=f"Enlarged View: {view_mode}", margin=dict(r=150))

            # Table Interaction
            st.subheader("Top Ranked Configurations")
            st.info("💡 **Tip**: Click a row in the table below to highlight that design in the heatmaps above.")
            
            display_df = rdf.sort_values("score", ascending=False).head(50).reset_index(drop=True)
            selection = st.dataframe(
                display_df.round(3),
                width="stretch",
                on_select="rerun",
                selection_mode="single-row",
                hide_index=True
            )

            if selection and selection.selection.rows:
                sel_idx = selection.selection.rows[0]
                sel_row = display_df.iloc[sel_idx]
                sx, sy = sel_row["width_um"], sel_row["U_m_s"]
                
                if view_mode == "All (2x2 Grid)":
                    for r in [1, 2]:
                        for c in [1, 2]:
                            fig_h.add_trace(go.Scatter(
                                x=[sx], y=[sy], mode='markers',
                                marker=dict(symbol='star', size=16, color='white', 
                                          line=dict(color='black', width=1.5)),
                                name="Selected Configuration", showlegend=False
                            ), row=r, col=c)
                else:
                    fig_h.add_trace(go.Scatter(
                        x=[sx], y=[sy], mode='markers',
                        marker=dict(symbol='star', size=20, color='white', 
                                  line=dict(color='black', width=2)),
                        showlegend=False
                    ))

            fig_h.update_layout(xaxis_title="Width (μm)", yaxis_title="Velocity (m/s)")
            st.plotly_chart(fig_h, width="stretch")
        else:
            st.info("Adjust the parameters on the left and click **Run Design Sweep** to generate the hyperparameter maps.")

    # --- TAB P: Particle Size Sweep ---
    with tab_p:
        st.header("Effect of Particle Size")
        col1, col2 = st.columns([1, 2])
        with col1:
            u_p = st.number_input("Mean Velocity (m/s)", 0.1, 5.0, 1.04, key="u_p_p")
            r_p = st.number_input("Radius (mm)", 1.0, 50.0, 4.3, key="r_p_p")
            dp_range = st.slider("Size range (μm)", 1.0, 40.0, (1.0, 25.0))
        
        with col2:
            df_p = sim.particle_sweep(u_p, r_p, dp_range[0], dp_range[1], model, alpha)
            fig_p = make_subplots(specs=[[{"secondary_y": True}]])
            fig_p.add_trace(go.Scatter(x=df_p["dp_um"], y=df_p["FL_pN"], name="Lift (pN)"), secondary_y=False)
            fig_p.add_trace(go.Scatter(x=df_p["dp_um"], y=df_p["FD_pN"], name="Drag (pN)"), secondary_y=False)
            fig_p.add_trace(go.Scatter(x=df_p["dp_um"], y=df_p["FL_over_FD"], name="Ratio (FL/FD)", line=dict(dash='dash')), secondary_y=True)
            
            fig_p.add_hline(y=1.0, line_dash="dot", secondary_y=True)
            fig_p.update_xaxes(title_text="dp (μm)", type="log")
            fig_p.update_yaxes(title_text="Force (pN)", type="log", secondary_y=False)
            fig_p.update_yaxes(title_text="Ratio (FL/FD)", secondary_y=True)
            fig_p.update_layout(title="Force Balance vs Particle Size", hovermode="x unified")
            st.plotly_chart(fig_p, width="stretch")
        
        xc = sim._find_crossover(df_p["dp_um"].to_numpy(), df_p["FL_over_FD"].to_numpy())
        if xc:
            st.success(f"**Crossover diameter:** {xc:.2f} μm")

    # --- TAB V: Velocity Sweep ---
    with tab_v:
        st.header("Effect of Fluid Velocity")
        col1, col2 = st.columns([1, 2])
        with col1:
            dp_v = st.number_input("Particle Diameter (μm)", 1.0, 50.0, 12.0, key="dp_v_v")
            r_v = st.number_input("Radius (mm) ", 1.0, 50.0, 4.3, key="r_v_v")
            u_range = st.slider("Velocity range (m/s) ", 0.01, 3.0, (0.1, 1.5))
        
        with col2:
            df_v = sim.velocity_sweep(dp_v, r_v, u_range[0], u_range[1], model, alpha)
            fig_v = make_subplots(specs=[[{"secondary_y": True}]])
            fig_v.add_trace(go.Scatter(x=df_v["U_m_s"], y=df_v["FL_pN"], name="Lift (pN)"), secondary_y=False)
            fig_v.add_trace(go.Scatter(x=df_v["U_m_s"], y=df_v["FD_pN"], name="Drag (pN)"), secondary_y=False)
            fig_v.add_trace(go.Scatter(x=df_v["U_m_s"], y=df_v["FL_over_FD"], name="Ratio (FL/FD)", line=dict(dash='dash')), secondary_y=True)
            
            fig_v.add_hline(y=1.0, line_dash="dot", secondary_y=True)
            fig_v.update_xaxes(title_text="Velocity (m/s)", type="log")
            fig_v.update_yaxes(title_text="Force (pN)", type="log", secondary_y=False)
            fig_v.update_yaxes(title_text="Ratio (FL/FD)", secondary_y=True)
            fig_v.update_layout(title="Focusing Ratio vs Velocity", hovermode="x unified")
            st.plotly_chart(fig_v, width="stretch")

    # --- TAB S: Spiral Decay Analysis ---
    with tab_s:
        st.header("Spiral Decay Analysis")
        col1, col2 = st.columns([1, 2])
        with col1:
            u_s = st.number_input("Velocity (m/s)  ", 0.1, 5.0, 1.04, key="u_s_s")
            dp_s = st.number_input("Particle Diameter (μm) ", 1.0, 50.0, 12.0, key="dp_s_s")
            r_range_s = st.slider("Radius range (mm)", 1.0, 50.0, (4.3, 15.0))
        
        with col2:
            df_s = sim.spiral_sweep(u_s, dp_s, r_range_s[0], r_range_s[1], model, alpha)
            fig_s = make_subplots(specs=[[{"secondary_y": True}]])
            fig_s.add_trace(go.Scatter(x=df_s["R_mm"], y=df_s["FL_pN"], name="Lift (pN)"), secondary_y=False)
            fig_s.add_trace(go.Scatter(x=df_s["R_mm"], y=df_s["FD_pN"], name="Drag (pN)"), secondary_y=False)
            fig_s.add_trace(go.Scatter(x=df_s["R_mm"], y=df_s["FL_over_FD"], name="Ratio (FL/FD)", line=dict(dash='dash')), secondary_y=True)
            
            fig_s.update_xaxes(title_text="Radius R (mm)")
            fig_s.update_yaxes(title_text="Force (pN)", secondary_y=False)
            fig_s.update_yaxes(title_text="Ratio (FL/FD)", secondary_y=True)
            fig_s.update_layout(title="Focusing Strength vs Curvature", hovermode="x unified")
            st.plotly_chart(fig_s, width="stretch")
        
        gain = df_s["FL_over_FD"].iloc[-1] / df_s["FL_over_FD"].iloc[0]
        st.info(f"**Focusing Ratio Multiplier:** {gain:.2f}x gain as radius increases.")

    # Technical Details
    with st.expander("Show Physics Model Context"):
        template_path = Path(__file__).parent / "TEMPLATE.md"
        if template_path.exists():
            text = template_path.read_text()
            if "## Physics Model" in text:
                physics_content = text.split("## Physics Model")[-1].split("---")[0]
                st.markdown("## Physics Model" + physics_content)
            else:
                st.markdown(text)

    with st.expander("Optimization & Scoring Framework"):
        st.markdown(r"""
        ### 1. Surrogate Design Metrics
        Dimensionless surrogates to describe performance:
        - **Transport ($U_D$)**: Mean Dean velocity along the spiral. Proxy for cross-interface transport.
        - **Focusing Ratio ($F_L/F_D$)**: Ratio of focusing lift to Dean-driven stirring.
            - **Outlet Ratio**: Focusing strength at final split. (Capped at 10.0 for scoring).
            - **Min Ratio**: Weakest focusing strength along path. (Capped at 5.0 for scoring).

        ### 2. Penalty Factors (Real-World Constraints)
        1. **Fabrication/Clogging ($P_{Fab}$)**: 
           - $w < 60 \mu m$: $P_{Fab} = 0.0$
           - $60 \le w < 100 \mu m$: $P_{Fab} = 0.2 + 0.8 \cdot \frac{w - 60}{40}$
           - $w \ge 100 \mu m$: $P_{Fab} = 1.0$
        2. **Laminar Penalty ($P_{Re}$)**: $Re > 1000$ (Turbulence/Mixing).
        3. **Shear Penalty ($P_{U}$)**: $U > 1.5$ m/s (Cell damage/Delamination).
        4. **Validity Penalty ($P_{De}$)**: Rezai model limit ($De > 30$).

        ### 3. Composite Design Score
        $$Score = (0.4 \cdot T_{norm} + 0.4 \cdot O_{norm} + 0.2 \cdot R_{norm}) \times P_{Fab} \times P_{Re} \times P_{Shear} \times P_{De} \times P_{Geom}$$
        Where:
        - $T_{norm}$ (Transport): Min-max normalized **Mean Dean Velocity** along the path.
        - $O_{norm}$ (Outlet): Normalized focusing strength at the **final exit**.
        - $R_{norm}$ (Robustness): Normalized focusing strength at the **weakest point** along the path.
        - $P_{Geom}$ (Geometric): Penalty if the device exceeds the **24mm slide limit** or the **2mm inlet limit**.
        """)

if __name__ == "__main__":
    run_gui()
