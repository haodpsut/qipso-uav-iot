"""Rotary-wing UAV propulsion power and trajectory energy.

Reference: Zeng, Y., Xu, J., Zhang, R., "Energy Minimization for Wireless
Communication With Rotary-Wing UAV," IEEE TWC, 2019.

Power as a function of instantaneous speed v (m/s):

    P(v) = P0 * (1 + 3 v^2 / U_tip^2)
         + Pi * sqrt( sqrt(1 + v^4 / (4 v0^4)) - v^2 / (2 v0^2) )
         + 0.5 * d0 * rho * s * A * v^3

Default constants match the Zeng-2019 reference rotary-wing parameters.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PropulsionParams:
    P0: float = 79.86          # blade profile power (W)
    Pi: float = 88.63          # induced power (W)
    U_tip: float = 120.0       # tip speed of the rotor blade (m/s)
    v0: float = 4.03           # mean rotor induced velocity in hover (m/s)
    d0: float = 0.6            # fuselage drag ratio
    rho: float = 1.225         # air density (kg/m^3)
    s: float = 0.05            # rotor solidity
    A: float = 0.503           # rotor disc area (m^2)
    P_hover: float = 168.49    # P0 + Pi for reference


def propulsion_power(v: torch.Tensor, params: PropulsionParams | None = None) -> torch.Tensor:
    """Per-element propulsion power (W). Works on any tensor shape."""
    p = params or PropulsionParams()
    v2 = v.pow(2)
    v4 = v2.pow(2)
    blade = p.P0 * (1.0 + 3.0 * v2 / (p.U_tip ** 2))
    inner = torch.sqrt(torch.clamp(1.0 + v4 / (4.0 * p.v0 ** 4), min=1e-12)) - v2 / (2.0 * p.v0 ** 2)
    induced = p.Pi * torch.sqrt(torch.clamp(inner, min=1e-12))
    parasite = 0.5 * p.d0 * p.rho * p.s * p.A * v2 * v.abs()
    return blade + induced + parasite


def trajectory_energy(trajectory: torch.Tensor, dt: float,
                      params: PropulsionParams | None = None) -> torch.Tensor:
    """Compute total propulsion energy (J) for a (..., N, 2) trajectory.

    trajectory[..., n, :] is the UAV planar coordinate at time slot n.
    dt is the slot duration (s).
    Returns a tensor of shape trajectory.shape[:-2] with energies.
    """
    diff = trajectory[..., 1:, :] - trajectory[..., :-1, :]
    v = diff.norm(dim=-1) / dt           # (..., N-1)
    p = propulsion_power(v, params)
    energy = (p * dt).sum(dim=-1)
    return energy
