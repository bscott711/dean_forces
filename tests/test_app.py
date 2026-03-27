import numpy as np

from dean_forces import DeanForcesSimulator, Geometry, DeanModel


def build():
    return DeanForcesSimulator(Geometry(width_um=150.0, height_um=150.0))


def test_dean_number_decreases_with_radius():
    sim = build()
    de_small_r = sim.dean_number(1.0, 4.3e-3)
    de_large_r = sim.dean_number(1.0, 9.5e-3)
    assert de_small_r > de_large_r


def test_simple_ud_decreases_with_radius():
    sim = build()
    ud_small_r = sim.dean_velocity(1.0, 4.3e-3, model=DeanModel.SIMPLE, alpha=0.3)
    ud_large_r = sim.dean_velocity(1.0, 9.5e-3, model=DeanModel.SIMPLE, alpha=0.3)
    assert ud_small_r > ud_large_r


def test_particle_ratio_increases_with_diameter():
    sim = build()
    df = sim.particle_sweep(
        u=1.04,
        r_mm=4.3,
        dp_start_um=1.0,
        dp_end_um=20.0,
        model=DeanModel.REZAI2017,
        alpha=0.3,
    )
    assert df["FL_over_FD"].iloc[-1] > df["FL_over_FD"].iloc[0]


def test_velocity_ratio_increases_with_velocity():
    sim = build()
    df = sim.velocity_sweep(
        dp_um=12.0,
        r_mm=4.3,
        u_start=0.01,
        u_end=1.0,
        model=DeanModel.REZAI2017,
        alpha=0.3,
    )
    assert df["FL_over_FD"].iloc[-1] > df["FL_over_FD"].iloc[0]


def test_rezai_relation_is_applied():
    sim = build()
    u = 1.0
    r_m = 4.3e-3
    de = sim.dean_number(u, r_m)
    ud = sim.dean_velocity(u, r_m, model=DeanModel.REZAI2017)
    lhs = ud * sim.g.s_m / sim.g.nu
    rhs = 0.031 * de**1.63
    assert np.isclose(lhs, rhs, rtol=1e-12)
