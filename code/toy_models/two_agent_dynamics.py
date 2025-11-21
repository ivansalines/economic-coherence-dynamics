#!/usr/bin/env python3
"""
Two-agent density–phase dynamics toy model.

This script implements the minimal two-subsystem model described in
"A Lagrangian Framework for Economic Coherence: Field Dynamics,
Phase Alignment, and Emergent Structure" (Ivan Salines, 2025).

The dynamical variables are:
    rho_A(t), rho_B(t), theta_A(t), theta_B(t)

with equations (in a spatially uniform setting):

    d rho_A / dt = - J * rho_A * rho_B * sin(theta_A - theta_B)
    d rho_B / dt = + J * rho_A * rho_B * sin(theta_A - theta_B)

    d theta_A / dt = V_A'(rho_A) + J * rho_B * [1 - cos(theta_A - theta_B)]
    d theta_B / dt = V_B'(rho_B) + J * rho_A * [1 - cos(theta_A - theta_B)]

We choose simple quadratic local potentials:

    V_i(rho_i) = 0.5 * k_i * (rho_i - rho0_i)^2

so that:

    V_i'(rho_i) = k_i * (rho_i - rho0_i)

This toy model is purely illustrative and is not calibrated to data.
It is meant to show:

    - conservation of total density: rho_A + rho_B = const
    - relaxation of the phase difference theta_A - theta_B toward 0
      when J > 0 and both densities remain positive
"""

import numpy as np
import matplotlib.pyplot as plt


class Params(object):
    def __init__(self, J=0.5, kA=0.2, kB=0.2, rho0A=1.0, rho0B=1.0):
        self.J = J
        self.kA = kA
        self.kB = kB
        self.rho0A = rho0A
        self.rho0B = rho0B


def potentials_prime(rho_A, rho_B, p):
    """
    Compute V_A'(rho_A) and V_B'(rho_B) for quadratic potentials:
        V_i(rho_i) = 0.5 * k_i * (rho_i - rho0_i)^2
        => V_i'(rho_i) = k_i * (rho_i - rho0_i)
    """
    VA_prime = p.kA * (rho_A - p.rho0A)
    VB_prime = p.kB * (rho_B - p.rho0B)
    return VA_prime, VB_prime


def rhs(t, y, p):
    """
    Right-hand side of the ODE system.

    y = [rho_A, rho_B, theta_A, theta_B]
    """
    rho_A, rho_B, theta_A, theta_B = y
    delta_theta = theta_A - theta_B

    # Derivatives of the local potentials
    VA_prime, VB_prime = potentials_prime(rho_A, rho_B, p)

    # Density equations
    drho_A = - p.J * rho_A * rho_B * np.sin(delta_theta)
    drho_B = + p.J * rho_A * rho_B * np.sin(delta_theta)

    # Phase equations
    dtheta_A = VA_prime + p.J * rho_B * (1.0 - np.cos(delta_theta))
    dtheta_B = VB_prime + p.J * rho_A * (1.0 - np.cos(delta_theta))

    return np.array([drho_A, drho_B, dtheta_A, dtheta_B])


def rk4_step(f, t, y, dt, p):
    """
    One Runge–Kutta 4 step:

        y_{n+1} = y_n + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    """
    k1 = f(t,           y,              p)
    k2 = f(t + 0.5*dt,  y + 0.5*dt*k1,  p)
    k3 = f(t + 0.5*dt,  y + 0.5*dt*k2,  p)
    k4 = f(t + dt,      y + dt*k3,      p)
    return y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def integrate_two_agent_model(
    y0,
    t_max=40.0,
    dt=0.01,
    params=None,
):
    """
    Integrate the two-agent system from t=0 to t=t_max with step dt.

    Returns:
        t:   array of times
        sol: array of shape (n_steps, 4) with [rho_A, rho_B, theta_A, theta_B]
    """
    if params is None:
        params = Params()

    n_steps = int(t_max / dt) + 1
    t = np.linspace(0.0, t_max, n_steps)
    sol = np.zeros((n_steps, 4), dtype=float)

    y = np.array(y0, dtype=float)
    sol[0] = y

    for i in range(1, n_steps):
        y = rk4_step(rhs, t[i-1], y, dt, params)
        sol[i] = y

    return t, sol


def main():
    # Parameters
    params = Params(
        J=0.5,
        kA=0.15,
        kB=0.20,
        rho0A=1.0,
        rho0B=1.0,
    )

    # Initial conditions:
    # slightly different densities and a non-zero phase mismatch
    rho_A0 = 1.2
    rho_B0 = 0.8
    theta_A0 = 0.0
    theta_B0 = 1.2  # ~69 degrees mismatch

    y0 = [rho_A0, rho_B0, theta_A0, theta_B0]

    t_max = 40.0
    dt = 0.02

    t, sol = integrate_two_agent_model(y0, t_max=t_max, dt=dt, params=params)
    rho_A = sol[:, 0]
    rho_B = sol[:, 1]
    theta_A = sol[:, 2]
    theta_B = sol[:, 3]
    delta_theta = np.unwrap(theta_A - theta_B)

    # Check approximate conservation of total density
    rho_tot = rho_A + rho_B
    print(f"Total density (initial): {rho_tot[0]:.6f}")
    print(f"Total density (final):   {rho_tot[-1]:.6f}")

    # Plot densities
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1 = axes[0]
    ax1.plot(t, rho_A, label=r"$\rho_A(t)$")
    ax1.plot(t, rho_B, label=r"$\rho_B(t)$", linestyle="--")
    ax1.set_ylabel("Density")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    # Plot phase difference
    ax2 = axes[1]
    ax2.plot(t, delta_theta)
    ax2.set_xlabel("Time")
    ax2.set_ylabel(r"$\Delta\theta(t)$")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Two-agent density–phase dynamics (toy model)", fontsize=12)
    fig.tight_layout(rect=[0, 0.0, 1, 0.95])

    plt.show()


if __name__ == "__main__":
    main()
