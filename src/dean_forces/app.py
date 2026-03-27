from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import typer
from streamlit.web import cli as stcli

app = typer.Typer(help="Simulate Dean-force trends in curved/spiral microchannels.", no_args_is_help=True)


class DeanModel(str, Enum):
    SIMPLE = "simple"
    OOKAWARA = "ookawara"
    REZAI2017 = "rezai2017"


@dataclass(frozen=True)
class Geometry:
    width_um: float = 150.0
    height_um: float = 150.0
    rho: float = 1000.0
    mu: float = 0.001

    def __post_init__(self) -> None:
        for name, value in {
            "width_um": self.width_um,
            "height_um": self.height_um,
            "rho": self.rho,
            "mu": self.mu,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be > 0")

    @property
    def width_m(self) -> float:
        return self.width_um * 1e-6

    @property
    def height_m(self) -> float:
        return self.height_um * 1e-6

    @property
    def dh_m(self) -> float:
        w = self.width_m
        h = self.height_m
        return 2.0 * w * h / (w + h)

    @property
    def s_m(self) -> float:
        return max(self.width_m, self.height_m)

    @property
    def nu(self) -> float:
        return self.mu / self.rho


class DeanForcesSimulator:
    def __init__(self, geometry: Geometry, outdir: str = "outputs") -> None:
        self.g = geometry
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _minmax(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        xmin = np.nanmin(x)
        xmax = np.nanmax(x)
        if np.isclose(xmax, xmin):
            return np.ones_like(x)
        return (x - xmin) / (xmax - xmin)

    def calculate_design_score(
        self,
        mean_ud_mm_s: np.ndarray,
        outlet_ratio: np.ndarray,
        min_ratio: np.ndarray,
        max_de: np.ndarray,
        u_vals: np.ndarray,
        dh_um: np.ndarray,
        width_um: np.ndarray,
        r_start_mm: float,
        r_end_mm: float,
        model: DeanModel,
        enforce_slide_limit: bool = True,
    ) -> np.ndarray:
        """Centralized scoring logic shared between CLI and GUI."""
        # Normalize metrics for ranking
        # Cap focusing ratios at a "sufficient" level (e.g., 10x and 5x) 
        # so that extreme values don't overwhelm the transport metrics.
        # Outlet = final performance, Min = robustness along the path.
        capped_outlet = np.minimum(outlet_ratio, 10.0)
        capped_min = np.minimum(min_ratio, 5.0)

        transport_norm = self._minmax(mean_ud_mm_s)
        outlet_focus_norm = self._minmax(np.log10(1.0 + capped_outlet))
        robustness_norm = self._minmax(np.log10(1.0 + capped_min))

        # Fabrication / clogging penalty based on width (User suggestion)
        # w_min = 60um (absolute printable), w_safe = 100um (reliable/clog-free)
        w_min = 60.0
        w_safe = 100.0
        p_low = 0.2
        
        P_fab = np.ones_like(width_um)
        P_fab[width_um < w_min] = 0.0 # Hard floor
        mask = (width_um >= w_min) & (width_um < w_safe)
        P_fab[mask] = p_low + (1.0 - p_low) * (width_um[mask] - w_min) / (w_safe - w_min)

        # Laminar flow penalty (Mixing occurs as Re -> 2000)
        re_vals = u_vals * (dh_um * 1e-6) * self.g.rho / self.g.mu
        laminar_penalty = np.where(re_vals > 1000, np.maximum(0.05, 1.0 - (re_vals - 1000) / 1000), 1.0)

        # Shear/Pressure penalty (Cells and delamination)
        shear_penalty = np.where(u_vals > 1.5, np.maximum(0.5, 1.0 - 0.5 * (u_vals - 1.5) / 1.5), 1.0)

        # Model validity penalty
        validity_penalty = np.ones_like(max_de)
        if model == DeanModel.REZAI2017:
            validity_penalty = np.minimum(1.0, 30.0 / max_de)

        # Geometric Constraints (User Request: Optional Slide Limit)
        # 1. Slide limit: 25mm slide, outer diameter (2*R_end) < 24mm (1mm buffer)
        P_geom = 1.0
        if enforce_slide_limit:
            if (2.0 * r_end_mm) > 24.0:
                # Soft decaying penalty if exceeded (User request: keep it gentle)
                P_geom *= np.exp(-0.25 * (2.0 * r_end_mm - 24.0)) 
            
            # 2. Inlet limit: 2x 1.5mm inlets + buffer implies R_start > 2.0mm
            if r_start_mm < 2.0:
                P_geom *= np.exp(-5.0 * (2.0 - r_start_mm)) 

        # Residence Time Penalty (Principled Decay)
        # t0: Comfortable threshold where we are still happy (s)
        # beta: How fast the penalty decays after t0
        t0 = 5.0    
        beta = 0.2  
        
        w_mm = width_um * 1e-3
        pitch_mm = 2.0 * w_mm 
        est_length_mm = np.pi * (r_end_mm**2 - r_start_mm**2) / pitch_mm
        res_time_s = (est_length_mm * 1e-3) / np.maximum(u_vals, 0.01)
        
        P_time = np.ones_like(res_time_s)
        mask = res_time_s > t0
        P_time[mask] = np.exp(-beta * (res_time_s[mask] - t0))

        # Composite score: product of normalized components and penalty factors
        return (
            0.40 * transport_norm
            + 0.40 * outlet_focus_norm
            + 0.20 * robustness_norm
        ) * validity_penalty * laminar_penalty * shear_penalty * P_fab * P_geom * P_time

    def design_sweep(
        self,
        dp_um: float,
        height_um: float,
        width_start_um: float,
        width_end_um: float,
        n_width: int,
        u_start: float,
        u_end: float,
        n_u: int,
        r_start_mm: float,
        r_end_mm: float,
        model: DeanModel,
        alpha: float,
        enforce_slide_limit: bool = True,
    ) -> pd.DataFrame:
        """Core sweep logic for design optimization (Single Source of Truth)."""
        widths = np.linspace(width_start_um, width_end_um, n_width)
        velocities = np.linspace(u_start, u_end, n_u)

        rows = []
        for w in widths:
            # We must recreate the geometry for each width in the sweep
            sweep_sim = DeanForcesSimulator(Geometry(width_um=w, height_um=height_um))
            for u in velocities:
                df = sweep_sim.spiral_sweep(
                    u=u,
                    dp_um=dp_um,
                    r_start_mm=r_start_mm,
                    r_end_mm=r_end_mm,
                    model=model,
                    alpha=alpha,
                    quiet=True,
                )

                # Estimate residence time for the report
                w_mm = w * 1e-3
                pitch_mm = 2.0 * w_mm
                est_L_mm = np.pi * (r_end_mm**2 - r_start_mm**2) / pitch_mm
                t_res = (est_L_mm * 1e-3) / max(u, 0.01)

                rows.append({
                    "width_um": w,
                    "height_um": height_um,
                    "Dh_um": sweep_sim.g.dh_m * 1e6,
                    "U_m_s": u,
                    "dp_um": dp_um,
                    "R_start_mm": r_start_mm,
                    "R_end_mm": r_end_mm,
                    "mean_Ud_mm_s": df["Ud_m_s"].mean() * 1e3,
                    "outlet_FL_over_FD": float(df["FL_over_FD"].iloc[-1]),
                    "min_FL_over_FD": float(df["FL_over_FD"].min()),
                    "max_De": float(df["De"].max()),
                    "res_time_s": t_res,
                })

        rank_df = pd.DataFrame(rows)
        rank_df["score"] = self.calculate_design_score(
            mean_ud_mm_s=rank_df["mean_Ud_mm_s"].to_numpy(),
            outlet_ratio=rank_df["outlet_FL_over_FD"].to_numpy(),
            min_ratio=rank_df["min_FL_over_FD"].to_numpy(),
            max_de=rank_df["max_De"].to_numpy(),
            u_vals=rank_df["U_m_s"].to_numpy(),
            dh_um=rank_df["Dh_um"].to_numpy(),
            width_um=rank_df["width_um"].to_numpy(),
            r_start_mm=r_start_mm,
            r_end_mm=r_end_mm,
            model=model,
            enforce_slide_limit=enforce_slide_limit,
        )
        
        # Normalize final scores to [0, 1] range for the current design space
        rank_df["score"] = self._minmax(rank_df["score"].to_numpy())
        
        return rank_df.sort_values("score", ascending=False).reset_index(drop=True)

    def reynolds(self, u: np.ndarray | float) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        return self.g.rho * u * self.g.dh_m / self.g.mu

    def dean_number(self, u: np.ndarray | float, r_m: np.ndarray | float) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        r_m = np.asarray(r_m, dtype=float)
        return self.reynolds(u) * np.sqrt(self.g.dh_m / (2.0 * r_m))

    def dean_velocity(
        self,
        u: np.ndarray | float,
        r_m: np.ndarray | float,
        model: DeanModel = DeanModel.REZAI2017,
        alpha: float = 0.30,
    ) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        r_m = np.asarray(r_m, dtype=float)
        de = self.dean_number(u, r_m)

        if model == DeanModel.SIMPLE:
            return alpha * u * np.sqrt(self.g.dh_m / (2.0 * r_m))

        if model == DeanModel.OOKAWARA:
            return 1.8e-4 * de**1.63

        if model == DeanModel.REZAI2017:
            return 0.031 * (self.g.nu / self.g.s_m) * de**1.63

        raise ValueError(f"Unknown model: {model}")

    def calc_forces(
        self,
        u: np.ndarray | float,
        r_m: np.ndarray | float,
        dp_m: np.ndarray | float,
        model: DeanModel = DeanModel.REZAI2017,
        alpha: float = 0.30,
        lift_prefactor: float = 1.5 * np.pi,
    ) -> dict[str, np.ndarray]:
        u = np.asarray(u, dtype=float)
        r_m = np.asarray(r_m, dtype=float)
        dp_m = np.asarray(dp_m, dtype=float)

        re = self.reynolds(u)
        de = self.dean_number(u, r_m)
        ud = self.dean_velocity(u, r_m, model=model, alpha=alpha)

        fl = lift_prefactor * self.g.rho * u**2 * dp_m**4 / self.g.dh_m**2
        fd = 3.0 * np.pi * self.g.mu * ud * dp_m

        return {
            "Re": re,
            "De": de,
            "Ud_m_s": ud,
            "FL_N": fl,
            "FD_N": fd,
            "FL_pN": fl * 1e12,
            "FD_pN": fd * 1e12,
            "FL_over_FD": fl / fd,
        }

    @staticmethod
    def _find_crossover(x: np.ndarray, ratio: np.ndarray) -> float | None:
        x = np.asarray(x, dtype=float)
        ratio = np.asarray(ratio, dtype=float)

        if np.nanmin(ratio) > 1.0 or np.nanmax(ratio) < 1.0:
            return None

        sign = np.sign(np.log10(ratio))
        idx = np.where(np.diff(sign))[0]
        if len(idx) == 0:
            return None

        i = idx[0]
        x0, x1 = x[i], x[i + 1]
        y0, y1 = np.log10(ratio[i]), np.log10(ratio[i + 1])
        return x0 + (0.0 - y0) * (x1 - x0) / (y1 - y0)

    def _warn_if_outside_rezai_range(self, de: np.ndarray, model: DeanModel, quiet: bool = False) -> str | None:
        if model == DeanModel.REZAI2017 and np.nanmax(de) > 30:
            msg = f"Warning: max De = {np.nanmax(de):.1f}; Rezai2017 was reported for De < 30."
            if not quiet:
                typer.secho(msg, fg=typer.colors.YELLOW)
            return msg
        return None

    def _save_df(self, df: pd.DataFrame, stem: str) -> Path:
        csv_path = self.outdir / f"{stem}.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    def _save_fig(self, fig: plt.Figure, stem: str) -> Path:
        png_path = self.outdir / f"{stem}.png"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return png_path

    def particle_sweep(
        self,
        u: float,
        r_mm: float,
        dp_start_um: float,
        dp_end_um: float,
        model: DeanModel,
        alpha: float,
        quiet: bool = False,
    ) -> pd.DataFrame:
        if u <= 0 or r_mm <= 0:
            raise ValueError("u and r_mm must be > 0")
        if dp_start_um <= 0 or dp_end_um <= 0 or dp_start_um >= dp_end_um:
            raise ValueError("Need 0 < dp_start_um < dp_end_um")

        dp_um = np.logspace(np.log10(dp_start_um), np.log10(dp_end_um), 200)
        dp_m = dp_um * 1e-6
        out = self.calc_forces(
            u=u,
            r_m=r_mm * 1e-3,
            dp_m=dp_m,
            model=model,
            alpha=alpha,
        )

        df = pd.DataFrame(
            {
                "dp_um": dp_um,
                **out,
            }
        )
        self._warn_if_outside_rezai_range(df["De"].to_numpy(), model, quiet=quiet)
        return df

    def velocity_sweep(
        self,
        dp_um: float,
        r_mm: float,
        u_start: float,
        u_end: float,
        model: DeanModel,
        alpha: float,
        quiet: bool = False,
    ) -> pd.DataFrame:
        if dp_um <= 0 or r_mm <= 0 or u_start <= 0 or u_end <= 0 or u_start >= u_end:
            raise ValueError("Require positive inputs and u_start < u_end")

        u = np.logspace(np.log10(u_start), np.log10(u_end), 200)
        out = self.calc_forces(
            u=u,
            r_m=r_mm * 1e-3,
            dp_m=dp_um * 1e-6,
            model=model,
            alpha=alpha,
        )

        df = pd.DataFrame({"U_m_s": u, **out})
        self._warn_if_outside_rezai_range(df["De"].to_numpy(), model, quiet=quiet)
        return df

    def spiral_sweep(
        self,
        u: float,
        dp_um: float,
        r_start_mm: float,
        r_end_mm: float,
        model: DeanModel,
        alpha: float,
        quiet: bool = False,
    ) -> pd.DataFrame:
        if u <= 0 or dp_um <= 0 or r_start_mm <= 0 or r_end_mm <= 0 or r_start_mm >= r_end_mm:
            raise ValueError("Require positive inputs and r_start_mm < r_end_mm")

        r_mm = np.linspace(r_start_mm, r_end_mm, 200)
        out = self.calc_forces(
            u=u,
            r_m=r_mm * 1e-3,
            dp_m=dp_um * 1e-6,
            model=model,
            alpha=alpha,
        )

        df = pd.DataFrame({"R_mm": r_mm, **out})
        self._warn_if_outside_rezai_range(df["De"].to_numpy(), model, quiet=quiet)
        return df

    def alpha_sweep(
        self,
        u: float,
        r_mm: float,
        dp_ref_um: float,
        dp_start_um: float,
        dp_end_um: float,
        alpha_start: float,
        alpha_end: float,
        n_alpha: int = 50,
    ) -> pd.DataFrame:
        if min(u, r_mm, dp_ref_um, dp_start_um, dp_end_um, alpha_start, alpha_end) <= 0:
            raise ValueError("All inputs must be > 0")
        if dp_start_um >= dp_end_um:
            raise ValueError("dp_start_um must be < dp_end_um")
        if alpha_start >= alpha_end:
            raise ValueError("alpha_start must be < alpha_end")

        alphas = np.linspace(alpha_start, alpha_end, n_alpha)
        dp_um = np.logspace(np.log10(dp_start_um), np.log10(dp_end_um), 300)
        dp_m = dp_um * 1e-6

        rows = []
        for alpha in alphas:
            out_range = self.calc_forces(
                u=u,
                r_m=r_mm * 1e-3,
                dp_m=dp_m,
                model=DeanModel.SIMPLE,
                alpha=alpha,
            )
            crossover = self._find_crossover(dp_um, out_range["FL_over_FD"])

            out_ref = self.calc_forces(
                u=u,
                r_m=r_mm * 1e-3,
                dp_m=dp_ref_um * 1e-6,
                model=DeanModel.SIMPLE,
                alpha=alpha,
            )

            rows.append(
                {
                    "alpha": alpha,
                    "crossover_dp_um": crossover,
                    "FL_over_FD_at_dp_ref": float(np.atleast_1d(out_ref["FL_over_FD"])[0]),
                    "Ud_mm_s": float(np.atleast_1d(out_ref["Ud_m_s"])[0]) * 1e3,
                    "De": float(np.atleast_1d(out_ref["De"])[0]),
                }
            )

        return pd.DataFrame(rows)

    def plot_particle(self, df: pd.DataFrame, stem: str, title: str) -> tuple[Path, Path]:
        """2-panel particle-size sweep: forces + FL/FD, clean layout."""
        fig, (ax_force, ax_ratio) = plt.subplots(
            1, 2, figsize=(11, 4), sharex=True
        )

        # Left: forces
        ax_force.loglog(df["dp_um"], df["FL_pN"], lw=2.5, color="tab:blue", label="Lift")
        ax_force.loglog(df["dp_um"], df["FD_pN"], lw=2.5, color="tab:red", label="Dean drag")
        ax_force.set_xlabel("Particle diameter (μm)")
        ax_force.set_ylabel("Force (pN)")
        ax_force.set_title("Forces")
        ax_force.grid(True, which="both", alpha=0.25)
        ax_force.legend(frameon=False, loc="lower right")

        # Right: ratio
        ax_ratio.semilogx(df["dp_um"], df["FL_over_FD"], lw=2.5, color="tab:green")
        ax_ratio.axhline(1.0, ls=":", lw=1.5, color="black", alpha=0.7)
        ax_ratio.set_xlabel("Particle diameter (μm)")
        ax_ratio.set_ylabel("Lift / Dean drag")
        ax_ratio.set_title("Focusing ratio")
        ax_ratio.grid(True, which="both", alpha=0.25)

        # Crossover marker
        x_cross = self._find_crossover(
            df["dp_um"].to_numpy(), df["FL_over_FD"].to_numpy()
        )
        if x_cross is not None:
            ax_ratio.axvline(x_cross, ls="--", lw=1.2, color="gray", alpha=0.8)
            ax_ratio.annotate(
                f"FL=FD at {x_cross:.2f} μm",
                xy=(x_cross, 1.0),
                xytext=(6, 10),
                textcoords="offset points",
                fontsize=9,
            )

        fig.suptitle(title, y=1.03)
        fig.tight_layout()

        png = self._save_fig(fig, stem)
        csv = self._save_df(df, stem)
        return png, csv

    def plot_velocity(self, df: pd.DataFrame, stem: str, title: str) -> tuple[Path, Path]:
        """2-panel velocity sweep, same visual structure as particle plot."""
        fig, (ax_force, ax_ratio) = plt.subplots(
            1, 2, figsize=(11, 4), sharex=False
        )

        # Left: forces
        ax_force.loglog(df["U_m_s"], df["FL_pN"], lw=2.5, color="tab:blue", label="Lift")
        ax_force.loglog(df["U_m_s"], df["FD_pN"], lw=2.5, color="tab:red", label="Dean drag")
        ax_force.set_xlabel("Mean axial velocity U (m/s)")
        ax_force.set_ylabel("Force (pN)")
        ax_force.set_title("Forces")
        ax_force.grid(True, which="both", alpha=0.25)
        ax_force.legend(frameon=False, loc="lower right")

        # Right: ratio
        ax_ratio.semilogx(df["U_m_s"], df["FL_over_FD"], lw=2.5, color="tab:green")
        ax_ratio.axhline(1.0, ls=":", lw=1.5, color="black", alpha=0.7)
        ax_ratio.set_xlabel("Mean axial velocity U (m/s)")
        ax_ratio.set_ylabel("Lift / Dean drag")
        ax_ratio.set_title("Focusing ratio")
        ax_ratio.grid(True, which="both", alpha=0.25)

        fig.suptitle(title, y=1.03)
        fig.tight_layout()

        png = self._save_fig(fig, stem)
        csv = self._save_df(df, stem)
        return png, csv

    def plot_spiral(self, df: pd.DataFrame, stem: str, title: str) -> tuple[Path, Path]:
        """2-panel spiral: forces vs R and De/Ud/ratio vs R, no overlapping legends."""
        fig, (ax_force, ax_dean) = plt.subplots(
            1, 2, figsize=(11, 4), sharex=True
        )

        # Left: forces along spiral
        ax_force.plot(df["R_mm"], df["FL_pN"], lw=2.5, color="tab:blue", label="Lift")
        ax_force.plot(df["R_mm"], df["FD_pN"], lw=2.5, color="tab:red", label="Dean drag")
        ax_force.set_xlabel("Radius of curvature R (mm)")
        ax_force.set_ylabel("Force (pN)")
        ax_force.set_title("Forces along spiral")
        ax_force.grid(True, alpha=0.25)
        ax_force.legend(frameon=False, loc="upper right")

        # Right: Dean metrics
        ax_dean.plot(df["R_mm"], df["De"], lw=2.2, color="purple", label="De")
        ax_dean.plot(
            df["R_mm"], df["Ud_m_s"] * 1e3, lw=2.2, color="tab:orange", label="Ud (mm/s)"
        )
        ax_dean.plot(
            df["R_mm"], df["FL_over_FD"], lw=2.2, color="tab:green", ls="--", label="Lift/drag"
        )
        ax_dean.set_xlabel("Radius of curvature R (mm)")
        ax_dean.set_ylabel("Dimensionless / velocity / ratio")
        ax_dean.set_title("Dean weakening & focusing")
        ax_dean.grid(True, alpha=0.25)
        ax_dean.legend(frameon=False, loc="upper right")

        fig.suptitle(title, y=1.03)
        fig.tight_layout()

        png = self._save_fig(fig, stem)
        csv = self._save_df(df, stem)
        return png, csv

    def plot_alpha_sweep(
        self, df: pd.DataFrame, stem: str, title: str, dp_ref_um: float
    ) -> tuple[Path, Path]:
        """2-panel alpha-sensitivity plot for the SIMPLE Ud model."""
        fig, (ax_cross, ax_ratio) = plt.subplots(
            1, 2, figsize=(11, 4), sharex=True
        )

        # Left: crossover diameter vs alpha
        ax_cross.plot(df["alpha"], df["crossover_dp_um"], lw=2.5, color="tab:blue")
        ax_cross.set_xlabel("α in Ud = α U √(Dh / 2R)")
        ax_cross.set_ylabel("Crossover diameter (μm)")
        ax_cross.set_title("Shift of FL=FD diameter")
        ax_cross.grid(True, alpha=0.25)

        # Right: FL/FD at reference size vs alpha
        ax_ratio.plot(df["alpha"], df["FL_over_FD_at_dp_ref"], lw=2.5, color="tab:green")
        ax_ratio.axhline(1.0, ls=":", lw=1.5, color="black", alpha=0.7)
        ax_ratio.set_xlabel("α in Ud = α U √(Dh / 2R)")
        ax_ratio.set_ylabel(f"FL/FD at {dp_ref_um:.1f} μm")
        ax_ratio.set_title("Focusing strength at target size")
        ax_ratio.grid(True, alpha=0.25)

        fig.suptitle(title, y=1.03)
        fig.tight_layout()

        png = self._save_fig(fig, stem)
        csv = self._save_df(df, stem)
        return png, csv


def build_sim(width_um: float, height_um: float) -> DeanForcesSimulator:
    return DeanForcesSimulator(Geometry(width_um=width_um, height_um=height_um))


def generate_markdown_report(
    sim: DeanForcesSimulator,
    stem: str,
    title: str,
    sections: list[str],
    u_ref: float,
    r_ref: float,
) -> Path:
    template_path = Path(__file__).parent / "TEMPLATE.md"
    if not template_path.exists():
        # Fallback if template is missing
        return sim.outdir / f"REPORT_{stem}.md"

    template = template_path.read_text()
    dh_um = sim.g.dh_m * 1e6

    report = (
        template.replace("{{TITLE}}", title)
        .replace("{{SECTIONS}}", "\n\n---\n\n".join(sections))
        .replace("{{RHO}}", f"{sim.g.rho}")
        .replace("{{MU}}", f"{sim.g.mu}")
        .replace("{{DH}}", f"{dh_um:.1f}")
        .replace("{{U_REF}}", f"{u_ref}")
        .replace("{{R_REF}}", f"{r_ref}")
    )

    report_path = sim.outdir / f"REPORT_{stem}.md"
    report_path.write_text(report)
    return report_path


@app.command()
def particle(
    u: Annotated[float, typer.Option()] = 1.04,
    width_um: Annotated[float, typer.Option()] = 150.0,
    height_um: Annotated[float, typer.Option()] = 150.0,
    r_mm: Annotated[float, typer.Option()] = 4.3,
    dp_start: Annotated[float, typer.Option()] = 1.0,
    dp_end: Annotated[float, typer.Option()] = 20.0,
    model: Annotated[DeanModel, typer.Option()] = DeanModel.REZAI2017,
    alpha: Annotated[float, typer.Option(help="Used only for model=simple")] = 0.30,
) -> None:
    sim = build_sim(width_um, height_um)
    df = sim.particle_sweep(u, r_mm, dp_start, dp_end, model, alpha)
    stem = f"particle_{model.value}_U{u}_W{width_um}_H{height_um}_R{r_mm}_dp{dp_start}-{dp_end}"
    png, csv = sim.plot_particle(df, stem, f"Particle sweep | model={model.value}")
    x_cross = sim._find_crossover(df["dp_um"].to_numpy(), df["FL_over_FD"].to_numpy())
    
    xc_str = f"**{x_cross:.2f} μm**" if x_cross else "no crossover found"
    section = f"""## 1. Particle Size Variation

Varying the particle diameter ($d_p$) while keeping velocity ($U$), hydraulic diameter ($D_h$), and radius ($R$) constant.

![Particle Size Simulation]({stem}.png)

- **Observation:** Lift force grows as $d_p^4$ while Drag grows as $d_p^1$.
- **Crossover:** At $U={u}$ m/s and $R={r_mm}$ mm, the forces balance ($F_L=F_D$) at approximately {xc_str}."""

    report_path = generate_markdown_report(sim, stem, "Dean Force Experiment: Particle Size", [section], u, r_mm)
    
    typer.echo(f"Saved: {png}")
    typer.echo(f"Report saved to: {report_path}")
    typer.echo(f"Re={df['Re'].iloc[0]:.2f}, De={df['De'].iloc[0]:.2f}, Ud={df['Ud_m_s'].iloc[0]:.5f} m/s")
    typer.echo(f"Crossover diameter: {x_cross:.3f} μm" if x_cross is not None else "No FL=FD crossover in range.")

@app.command()
def velocity(
    dp_um: Annotated[float, typer.Option()] = 12.0,
    width_um: Annotated[float, typer.Option()] = 150.0,
    height_um: Annotated[float, typer.Option()] = 150.0,
    r_mm: Annotated[float, typer.Option()] = 4.3,
    u_start: Annotated[float, typer.Option()] = 0.01,
    u_end: Annotated[float, typer.Option()] = 1.2,
    model: Annotated[DeanModel, typer.Option()] = DeanModel.REZAI2017,
    alpha: Annotated[float, typer.Option(help="Used only for model=simple")] = 0.30,
) -> None:
    sim = build_sim(width_um, height_um)
    df = sim.velocity_sweep(dp_um, r_mm, u_start, u_end, model, alpha)
    stem = f"velocity_{model.value}_dp{dp_um}_W{width_um}_H{height_um}_R{r_mm}_U{u_start}-{u_end}"
    png, csv = sim.plot_velocity(df, stem, f"Velocity sweep | model={model.value}")
    x_cross = sim._find_crossover(df["U_m_s"].to_numpy(), df["FL_over_FD"].to_numpy())
    
    xc_str = f"**{x_cross:.2f} m/s**" if x_cross else "no crossover found"
    section = f"""## 1. Velocity Variation

Analyzing how fluid velocity ($U$) affects the force balance for a fixed {dp_um}μm particle.

![Velocity Simulation]({stem}.png)

- **Observation:** Both forces increase with velocity, but Lift ($U^2$) grows faster than Drag ($U^1$).
- **Crossover:** The critical velocity required for $F_L > F_D$ for a {dp_um}μm particle is {xc_str}."""

    report_path = generate_markdown_report(sim, stem, "Dean Force Experiment: Velocity", [section], u_start, r_mm)
    
    typer.echo(f"Saved: {png}")
    typer.echo(f"Report saved to: {report_path}")

@app.command()
def spiral(
    u: Annotated[float, typer.Option()] = 1.04,
    dp_um: Annotated[float, typer.Option()] = 12.0,
    width_um: Annotated[float, typer.Option()] = 150.0,
    height_um: Annotated[float, typer.Option()] = 150.0,
    r_start_mm: Annotated[float, typer.Option()] = 4.3,
    r_end_mm: Annotated[float, typer.Option()] = 9.5,
    model: Annotated[DeanModel, typer.Option()] = DeanModel.REZAI2017,
    alpha: Annotated[float, typer.Option(help="Used only for model=simple")] = 0.30,
) -> None:
    sim = build_sim(width_um, height_um)
    df = sim.spiral_sweep(u, dp_um, r_start_mm, r_end_mm, model, alpha)
    stem = f"spiral_{model.value}_dp{dp_um}_W{width_um}_H{height_um}_R{r_start_mm}-{r_end_mm}_U{u}"
    png, csv = sim.plot_spiral(df, stem, f"Spiral sweep | model={model.value}")
    ratio_start = df["FL_over_FD"].iloc[0]
    ratio_end = df["FL_over_FD"].iloc[-1]
    ratio_increase = ratio_end / ratio_start

    section = f"""## 1. Spiral Decay (Varying Radius)

Simulating a particle traveling outward in a spiral, where the radius of curvature ($R$) increases over time.

![Spiral Decay Simulation]({stem}.png)

- **Observation:** As $R$ increases from {r_start_mm} mm to {r_end_mm} mm, the Dean Drag decreases because $U_D \\propto 1/\\sqrt{{R}}$.
- **Result:** The $F_L/F_D$ ratio increases by **{ratio_increase:.2f}x**, strengthening the focusing effect as the particle moves outward."""

    report_path = generate_markdown_report(sim, stem, "Dean Force Experiment: Spiral Decay", [section], u, r_start_mm)
    
    typer.echo(f"Saved: {png}")
    typer.echo(f"Report saved to: {report_path}")

@app.command("alpha-sweep")
def alpha_sweep(
    u: Annotated[float, typer.Option()] = 1.04,
    width_um: Annotated[float, typer.Option()] = 150.0,
    height_um: Annotated[float, typer.Option()] = 150.0,
    r_mm: Annotated[float, typer.Option()] = 4.3,
    dp_ref_um: Annotated[float, typer.Option()] = 12.0,
    dp_start: Annotated[float, typer.Option()] = 1.0,
    dp_end: Annotated[float, typer.Option()] = 25.0,
    alpha_start: Annotated[float, typer.Option()] = 0.05,
    alpha_end: Annotated[float, typer.Option()] = 0.60,
    n_alpha: Annotated[int, typer.Option()] = 50,
) -> None:
    sim = build_sim(width_um, height_um)
    df = sim.alpha_sweep(u, r_mm, dp_ref_um, dp_start, dp_end, alpha_start, alpha_end, n_alpha)
    stem = f"alpha_sweep_U{u}_W{width_um}_H{height_um}_R{r_mm}_dpref{dp_ref_um}"
    png, csv = sim.plot_alpha_sweep(df, stem, "Sensitivity to simple-model alpha", dp_ref_um)
    
    section = f"""## Alpha Sensitivity

Analyzing how the calibration parameter $\\alpha$ affect the crossover diameter.

![Alpha Sweep]({stem}.png)"""
    report_path = generate_markdown_report(sim, stem, "Dean Force Experiment: Alpha Sweep", [section], u, r_mm)
    
    typer.echo(f"Saved: {png}")
    typer.echo(f"Report saved to: {report_path}")

@app.command("all")
def run_all(
    u: Annotated[float, typer.Option()] = 1.04,
    dp_um: Annotated[float, typer.Option()] = 12.0,
    width_um: Annotated[float, typer.Option()] = 150.0,
    height_um: Annotated[float, typer.Option()] = 150.0,
    r_mm: Annotated[float, typer.Option()] = 4.3,
    r_start_mm: Annotated[float, typer.Option()] = 4.3,
    r_end_mm: Annotated[float, typer.Option()] = 9.5,
    dp_start: Annotated[float, typer.Option()] = 1.0,
    dp_end: Annotated[float, typer.Option()] = 20.0,
    u_start: Annotated[float, typer.Option()] = 0.01,
    u_end: Annotated[float, typer.Option()] = 1.2,
    model: Annotated[DeanModel, typer.Option()] = DeanModel.REZAI2017,
    alpha: Annotated[float, typer.Option()] = 0.30,
) -> None:
    sim = build_sim(width_um, height_um)
    dh_um = sim.g.dh_m * 1e6
    stem_all = f"all_{model.value}_U{u}_R{r_mm}"

    # 1. Particle sweep
    df_p = sim.particle_sweep(u, r_mm, dp_start, dp_end, model, alpha)
    stem_p = f"all_particle_{model.value}"
    sim.plot_particle(df_p, stem_p, f"Particle sweep | model={model.value}")
    x_cross_p = sim._find_crossover(df_p["dp_um"].to_numpy(), df_p["FL_over_FD"].to_numpy())
    xcp_str = f"**{x_cross_p:.2f} μm**" if x_cross_p else "no crossover found"
    section_p = f"""## 1. Particle Size Variation

![Particle Size]({stem_p}.png)

- **Crossover:** At $U={u}$ m/s and $R={r_mm}$ mm, $F_L=F_D$ at approximately {xcp_str}."""

    # 2. Spiral sweep
    df_s = sim.spiral_sweep(u, dp_um, r_start_mm, r_end_mm, model, alpha)
    stem_s = f"all_spiral_{model.value}"
    sim.plot_spiral(df_s, stem_s, f"Spiral sweep | model={model.value}")
    ratio_increase = df_s["FL_over_FD"].iloc[-1] / df_s["FL_over_FD"].iloc[0]
    section_s = f"""## 2. Spiral Decay

![Spiral Decay]({stem_s}.png)

- **Result:** The $F_L/F_D$ ratio increases by **{ratio_increase:.2f}x** from $R={r_start_mm}$ to {r_end_mm} mm."""

    # 3. Velocity sweep
    df_v = sim.velocity_sweep(dp_um, r_mm, u_start, u_end, model, alpha)
    stem_v = f"all_velocity_{model.value}"
    sim.plot_velocity(df_v, stem_v, f"Velocity sweep | model={model.value}")
    x_cross_v = sim._find_crossover(df_v["U_m_s"].to_numpy(), df_v["FL_over_FD"].to_numpy())
    xcv_str = f"**{x_cross_v:.2f} m/s**" if x_cross_v else "no crossover found"
    section_v = f"""## 3. Velocity Variation

![Velocity]({stem_v}.png)

- **Crossover:** For a {dp_um}μm particle, $F_L=F_D$ at {xcv_str}."""

    report_path = generate_markdown_report(sim, stem_all, "Comprehensive Dean Force Study", [section_p, section_s, section_v], u, r_mm)
    typer.echo(f"All simulations complete. Report saved to {report_path}")


@app.command("design-heatmap")
def design_heatmap(
    dp_um: Annotated[float, typer.Option(help="Target HL60 diameter (μm)")] = 12.0,
    height_um: Annotated[float, typer.Option(help="Fixed channel height (μm)")] = 150.0,
    width_start_um: Annotated[float, typer.Option(help="Min channel width (μm)")] = 75.0,
    width_end_um: Annotated[float, typer.Option(help="Max channel width (μm)")] = 250.0,
    n_width: Annotated[int, typer.Option(help="Number of width samples")] = 100,
    u_start: Annotated[float, typer.Option(help="Min velocity (m/s)")] = 0.10,
    u_end: Annotated[float, typer.Option(help="Max velocity (m/s)")] = 1.50,
    n_u: Annotated[int, typer.Option(help="Number of velocity samples")] = 100,
    r_start_mm: Annotated[float, typer.Option(help="Spiral start radius (mm)")] = 2.0,
    r_end_mm: Annotated[float, typer.Option(help="Spiral end radius (mm)")] = 15.0,
    model: Annotated[DeanModel, typer.Option()] = DeanModel.REZAI2017,
    alpha: Annotated[float, typer.Option(help="Used only for model=simple")] = 0.30,
    cmap: Annotated[str, typer.Option(help="Colormap for heatmaps")] = "viridis",
    top_n: Annotated[int, typer.Option(help="Rows to print from ranking")] = 15,
) -> None:
    if width_start_um <= 0 or width_end_um <= 0 or width_start_um >= width_end_um:
        raise ValueError("Require 0 < width_start_um < width_end_um")
    if u_start <= 0 or u_end <= 0 or u_start >= u_end:
        raise ValueError("Require 0 < u_start < u_end")
    if r_start_mm <= 0 or r_end_mm <= 0 or r_start_mm >= r_end_mm:
        raise ValueError("Require 0 < r_start_mm < r_end_mm")
    if dp_um <= 0:
        raise ValueError("dp_um must be > 0")

    sim = build_sim(width_um=width_start_um, height_um=height_um)
    rank_df = sim.design_sweep(
        dp_um=dp_um,
        height_um=height_um,
        width_start_um=width_start_um,
        width_end_um=width_end_um,
        n_width=n_width,
        u_start=u_start,
        u_end=u_end,
        n_u=n_u,
        r_start_mm=r_start_mm,
        r_end_mm=r_end_mm,
        model=model,
        alpha=alpha,
    )

    rank_df = rank_df.sort_values("score", ascending=False).reset_index(drop=True)

    # Save ranking table
    outdir = Path("outputs")
    outdir.mkdir(exist_ok=True)
    stem = (
        f"design_heatmap_{model.value}_dp{dp_um}_H{height_um}"
        f"_W{width_start_um}-{width_end_um}_U{u_start}-{u_end}"
        f"_R{r_start_mm}-{r_end_mm}"
    )
    csv_path = outdir / f"{stem}.csv"
    rank_df.to_csv(csv_path, index=False)

    # Pivot for heatmaps
    score_grid = rank_df.pivot(index="U_m_s", columns="width_um", values="score").sort_index()
    ud_grid = rank_df.pivot(index="U_m_s", columns="width_um", values="mean_Ud_mm_s").sort_index()
    ratio_grid = rank_df.pivot(index="U_m_s", columns="width_um", values="outlet_FL_over_FD").sort_index()
    de_grid = rank_df.pivot(index="U_m_s", columns="width_um", values="max_De").sort_index()

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax1, ax2, ax3, ax4 = axes.flat

    im1 = ax1.imshow(
        score_grid.values,
        origin="lower",
        aspect="auto",
        extent=[score_grid.columns.min(), score_grid.columns.max(), score_grid.index.min(), score_grid.index.max()],
        cmap=cmap,
    )
    ax1.set_title("Composite design score")
    ax1.set_xlabel("Channel width (μm)")
    ax1.set_ylabel("Velocity U (m/s)")
    fig.colorbar(im1, ax=ax1, shrink=0.9)

    im2 = ax2.imshow(
        ud_grid.values,
        origin="lower",
        aspect="auto",
        extent=[ud_grid.columns.min(), ud_grid.columns.max(), ud_grid.index.min(), ud_grid.index.max()],
        cmap=cmap,
    )
    ax2.set_title("Mean Dean velocity (mm/s)")
    ax2.set_xlabel("Channel width (μm)")
    ax2.set_ylabel("Velocity U (m/s)")
    fig.colorbar(im2, ax=ax2, shrink=0.9)

    im3 = ax3.imshow(
        ratio_grid.values,
        origin="lower",
        aspect="auto",
        extent=[ratio_grid.columns.min(), ratio_grid.columns.max(), ratio_grid.index.min(), ratio_grid.index.max()],
        cmap=cmap,
    )
    ax3.set_title("Outlet FL/FD")
    ax3.set_xlabel("Channel width (μm)")
    ax3.set_ylabel("Velocity U (m/s)")
    fig.colorbar(im3, ax=ax3, shrink=0.9)

    im4 = ax4.imshow(
        de_grid.values,
        origin="lower",
        aspect="auto",
        extent=[de_grid.columns.min(), de_grid.columns.max(), de_grid.index.min(), de_grid.index.max()],
        cmap=cmap,
    )
    ax4.set_title("Max Dean number")
    ax4.set_xlabel("Channel width (μm)")
    ax4.set_ylabel("Velocity U (m/s)")
    fig.colorbar(im4, ax=ax4, shrink=0.9)

    fig.suptitle(
        f"HL60 design heatmap | dp={dp_um:.1f} μm, h={height_um:.1f} μm, "
        f"R={r_start_mm:.1f}-{r_end_mm:.1f} mm, model={model.value}",
        y=1.02,
    )

    png_path = outdir / f"{stem}.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    typer.echo(f"Saved heatmap: {png_path}")
    typer.echo(f"Saved ranking CSV: {csv_path}")
    typer.echo("")
    typer.echo("Top-ranked designs:")
    typer.echo(
        rank_df[
            [
                "score",
                "width_um",
                "Dh_um",
                "U_m_s",
                "mean_Ud_mm_s",
                "outlet_FL_over_FD",
                "min_FL_over_FD",
                "max_De",
            ]
        ]
        .head(top_n)
        .round(3)
        .to_string(index=False)
    )


@app.command()
def gui():
    """Launch the interactive Streamlit dashboard."""
    gui_script = Path(__file__).parent / "gui.py"
    if not gui_script.exists():
        typer.echo(f"Error: GUI script not found at {gui_script}")
        raise typer.Exit(code=1)
    
    sys.argv = ["streamlit", "run", str(gui_script)]
    sys.exit(stcli.main())


if __name__ == "__main__":
    app()
