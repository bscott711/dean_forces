# Dean Microfluidic Force Simulations

A command-line tool for simulating and visualizing the balance between **Inertial Lift ($F_L$)** and **Dean Drag ($F_D$)** in spiral microfluidic channels. This balance is critical for size-based particle separation and focusing.

## Features

- **Multiple Scaling Models**: Switch between a simple square-root scaling and the semi-empirical **Rezai 2017** correlation for Dean velocity.
- **Particle Size Sweeps**: Identify the crossover diameter ($F_L = F_D$) for any geometry.
- **Velocity Sweeps**: Determine critical focusing speeds for target particle sizes.
- **Spiral Decay Simulation**: Model how focusing strength $F_L/F_D$ changes as a particle travels outward in a spiral ($R$ increases).
- **Design Optimization Heatmap**: Sweep channel width and velocity simultaneously to find the "optimal ridge" for cell focusing.
- **Composite Scoring Framework**: Balance **Transport**, **Focusing**, and **Robustness** with high-fidelity physical penalties.
- **Interactive GUI**: Linked zooming, star-highlighting for top configurations, and real-time parameter exploration.
- **Automated Reporting**: Generates unique, parameter-aware Markdown reports with embedded plots for every simulation run.
- **CSV Exports**: Saves raw simulation data alongside visual plots for downstream analysis.

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast and reliable dependency management.

```bash
# Clone the repository
git clone <repo-url>
cd dean_forces

# Initialize virtual environment and install dependencies
uv sync
```

## Usage

The tool is available via the `dean-force` entry point. Use `--help` to see all available subcommands.

```bash
# Run a comprehensive study (Particle, Velocity, and Spiral sweeps)
uv run dean-force all

# Launch the interactive Streamlit dashboard
uv run dean-force gui

# Run a specific particle size sweep
uv run dean-force particle --u 1.2 --r-mm 5.0 --dp-start 5 --dp-end 25

# Analyze sensitivity to the alpha calibration parameter
uv run dean-force alpha-sweep --dp-ref-um 12.0
```

### Main Subcommands

- `design-heatmap`: Generate a 2D map of the design space (Width vs Velocity) with composite scoring.
- `particle`: Sweep particle diameters at fixed fluid velocity and curvature.
- `velocity`: Sweep mean axial velocity for a fixed particle size.
- `spiral`: Simulate outward travel in a spiral (varying $R$).
- `alpha-sweep`: Explore the sensitivity of the simple model to its calibration constant.
- `all`: Generate a single report for a complete experimental profile.
- `gui`: Launch the full interactive Streamlit dashboard.

## Physics Overview

The simulation calculates the net focusing behavior governed by the force ratio:

$$
\frac{F_L}{F_D} \propto U^{0.37} d_p^3 R^{0.815} \quad (\text{Rezai 2017 model})
$$

Where:

- $U$ is the mean axial velocity.
- $d_p$ is the particle diameter.
- $R$ is the radius of curvature.
- $D_h$ is the hydraulic diameter.

## Design Optimization Logic

The optimization routine uses a weighted score adjusted by **four critical physical penalties**:

1. **Fabrication/Clogging ($P_{Fab}$)**: Penalizes channel widths below $100\mu m$ (hard floor at $60\mu m$).
2. **Laminar Limit ($P_{Re}$)**: Prevents mixing by penalizing flows with $Re > 1000$.
3. **Shear/Damage ($P_{U}$)**: Limits cell damage and delamination ($U > 1.5$ m/s).
4. **Model Validity ($P_{De}$)**: Applies a reciprocal penalty for using the Rezai model beyond its reported limit ($De > 30$).

The final **Composite Score** is calculated as:
$$Score = (0.40 \cdot T_{norm} + 0.40 \cdot O_{norm} + 0.20 \cdot R_{norm}) \times P_{Fab} \times P_{Re} \times P_{Shear} \times P_{De}$$

Where $T$ is Transport, $O$ is Outlet Focusing, and $R$ is Path Robustness.

## Development

Run unit tests with `pytest`:

```bash
uv run pytest tests/
```

Tests cover:

- Dean number and velocity scaling.
- Crossover trend validation.
- Numerical stability of the semi-empirical model.
