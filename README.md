# Dean Microfluidic Force Simulations

A command-line tool for simulating and visualizing the balance between **Inertial Lift ($F_L$)** and **Dean Drag ($F_D$)** in spiral microfluidic channels. This balance is critical for size-based particle separation and focusing.

## Features

- **Multiple Scaling Models**: Switch between a simple square-root scaling and the semi-empirical **Rezai 2017** correlation for Dean velocity.
- **Particle Size Sweeps**: Identify the crossover diameter ($F_L = F_D$) for any geometry.
- **Velocity Sweeps**: Determine critical focusing speeds for target particle sizes.
- **Spiral Decay Simulation**: Model how focusing strength $F_L/F_D$ changes as a particle travels outward in a spiral ($R$ increases).
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

# Run a specific particle size sweep
uv run dean-force particle --u 1.2 --r-mm 5.0 --dp-start 5 --dp-end 25

# Analyze sensitivity to the alpha calibration parameter
uv run dean-force alpha-sweep --dp-ref-um 12.0
```

### Main Subcommands

- `particle`: Sweep particle diameters at fixed fluid velocity and curvature.
- `velocity`: Sweep mean axial velocity for a fixed particle size.
- `spiral`: Simulate outward travel in a spiral (varying $R$).
- `alpha-sweep`: Explore the sensitivity of the simple model to its calibration constant.
- `all`: Generate a single report for a complete experimental profile.

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

Detailed scaling laws and model assumptions are automatically included in the `REPORT.md` generated with each run.

## Development

Run unit tests with `pytest`:

```bash
uv run pytest tests/
```

Tests cover:

- Dean number and velocity scaling.
- Crossover trend validation.
- Numerical stability of the semi-empirical model.
