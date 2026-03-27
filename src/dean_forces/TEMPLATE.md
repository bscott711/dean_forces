# {{TITLE}}

## Physics Model

We use two related descriptions of the Dean secondary flow: a **simple scaling model** (useful for intuition) and a **semi‑empirical model** (used in the simulations below).

### 1. Inertial lift

For a neutrally buoyant spherical particle in a rectangular channel, the inertial lift force is modeled as

$$
F_L \sim C_L \,\rho\, U^2 \frac{d_p^4}{D_h^2},
$$

where $C_L$ is an order‑unity coefficient, $\rho$ is the fluid density, $U$ the mean axial velocity, $d_p$ the particle diameter, and $D_h$ the hydraulic diameter.

### 2. Simple Dean scaling (for intuition)

If we approximate the Dean secondary velocity as

$$
U_D \propto U \sqrt{\frac{D_h}{2R}},
$$

then the Dean drag on the particle is

$$
F_D = 3\pi\mu U_D d_p \propto \mu\, U\, d_p\,\sqrt{\frac{D_h}{2R}}.
$$

Combining these expressions, the force ratio scales as

$$
\frac{F_L}{F_D}
\;\propto\;
\frac{\rho U^2 d_p^4 / D_h^2}{\mu U d_p \sqrt{D_h/(2R)}}
\;\propto\;
\frac{\rho}{\mu}\, U\, d_p^3\, R^{1/2}\, D_h^{-5/2}.
$$

This **simple model** is handy for back‑of‑the‑envelope reasoning: increasing $U$, $d_p$, or $R$ all promote focusing (larger $F_L/F_D$) at fixed geometry.

### 3. Semi‑empirical Dean model (used in code)

The simulations, however, use the **Rezai 2017** correlation for the average Dean velocity in curved microchannels:

$$
U_D = 0.031 \left(\frac{\nu}{s}\right) De^{1.63},
$$

where $\nu=\mu/\rho$ is the kinematic viscosity, $s=\max(w,h)$ is the larger channel dimension, and

$$
De = Re \sqrt{\frac{D_h}{2R}}, \qquad Re = \frac{\rho U D_h}{\mu}.
$$

Substituting this $U_D$ into the drag expression $F_D = 3\pi\mu U_D d_p$ gives

$$
F_D \propto \mu \left(\frac{\nu}{s}\right) De^{1.63} d_p
      \propto \rho^{0.63}\mu^{0.37} U^{1.63} d_p\, D_h^{0.815}\, R^{-0.815}\, s^{-1}.
$$

The resulting lift‑to‑drag ratio scales as

$$
\frac{F_L}{F_D}
\;\propto\;
\frac{\rho U^2 d_p^4 / D_h^2}{\mu (\nu/s) De^{1.63} d_p}
\;\propto\;
U^{0.37} d_p^3 R^{0.815} \times (\text{geometry, fluid constants}),
$$

so in the **Rezai2017 model** the dependence on velocity is weaker ($\propto U^{0.37}$) and the dependence on curvature radius is stronger ($\propto R^{0.815}$) than in the simple square‑root scaling.

In the code, you can switch between these two views via the `--model` flag:

- `--model simple` uses $U_D = \alpha U \sqrt{D_h/(2R)}$ with tunable $\alpha$ (explored by `alpha-sweep`).
- `--model rezai2017` uses the semi‑empirical correlation above and is the default for the figures produced by:

```bash
dean-force particle  --u 1.04 --width-um 150 --height-um 150 --r-mm 4.3 --dp-start 1 --dp-end 20 --model rezai2017
dean-force velocity  --dp-um 12 --width-um 150 --height-um 150 --r-mm 4.3 --u-start 0.05 --u-end 1.2 --model rezai2017
dean-force spiral    --u 1.04 --dp-um 12 --width-um 150 --height-um 150 --r-start-mm 4.3 --r-end-mm 9.5 --model rezai2017
dean-force alpha-sweep --u 1.04 --width-um 150 --height-um 150 --r-mm 4.3 --dp-ref-um 12 --alpha-start 0.05 --alpha-end 0.6
```

---

{{SECTIONS}}

---

## Summary of Parameters

| Parameter | Value |
| :--- | :--- |
| Fluid Density ($\rho$) | {{RHO}} kg/m³ |
| Dynamic Viscosity ($\mu$) | {{MU}} Pa·s |
| Hydraulic Diameter ($D_h$) | {{DH}} μm |
| Reference Velocity ($U$) | {{U_REF}} m/s |
| Reference Radius ($R$) | {{R_REF}} mm |
