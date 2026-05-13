"""UAV-IoT trajectory optimization environment.

Single-UAV scenario: K IoT nodes at fixed positions, one UAV flies at altitude H
across N time slots of duration dt seconds. Decision variable is the trajectory
q[1..N] in R^2 (start q[0] and end q[N+1] are fixed).

Fitness combines:
  - propulsion energy   (J)         -> minimize
  - data-shortfall      (Mbits)     -> penalty if not collected by mission end
  - velocity violation              -> penalty if |q[n]-q[n-1]| > V_max * dt

The whole environment is batched on a torch tensor of shape
(B, N, 2): B = swarm size or batched runs. Designed to run on CUDA.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .channel import ChannelParams, achievable_rate_bps
from .energy_model import PropulsionParams, trajectory_energy


@dataclass
class Scenario:
    area: float = 1000.0                      # meters (square side)
    K: int = 20                               # number of IoT nodes
    H: float = 100.0                          # UAV altitude (m)
    N: int = 80                               # number of free trajectory waypoints
    dt: float = 1.0                           # slot duration (s)
    V_max: float = 30.0                       # m/s
    data_per_node_mbits: tuple = (10.0, 100.0)
    start: tuple = (0.0, 0.0)
    end: tuple = (1000.0, 1000.0)
    energy_budget_j: float = 1.5e5            # mission energy budget
    service_radius_m: float = 200.0           # horiz range to be "in service"
    seed: int = 0
    # multi-UAV
    M: int = 1                                # number of UAVs
    # penalty weights
    w_data: float = 50.0                      # J per Mbit not delivered
    w_vel: float = 1e3                        # J per m/s overshoot squared
    device: str = "cpu"


class UAVIoTEnv:
    """Builds a scenario and exposes a batched fitness function."""

    def __init__(self, scenario: Scenario,
                 prop: PropulsionParams | None = None,
                 chan: ChannelParams | None = None):
        self.sc = scenario
        self.prop = prop or PropulsionParams()
        self.chan = chan or ChannelParams()
        self.device = torch.device(scenario.device)

        g = torch.Generator(device="cpu").manual_seed(scenario.seed)
        nodes = torch.rand(scenario.K, 2, generator=g) * scenario.area
        d_lo, d_hi = scenario.data_per_node_mbits
        data = torch.rand(scenario.K, generator=g) * (d_hi - d_lo) + d_lo

        self.nodes = nodes.to(self.device)
        self.data_required = data.to(self.device)        # Mbits
        self.start = torch.tensor(scenario.start, device=self.device)
        self.end = torch.tensor(scenario.end, device=self.device)
        self.dim = scenario.N * 2

    # ---------- helpers ----------
    def warm_start(self, S: int, noise_frac: float = 0.01,
                   seed: int | None = None) -> torch.Tensor:
        """Initialize S particles around the straight-line trajectory from
        start to end (+ Gaussian noise). Gives every optimizer a feasible
        starting region rather than uniform random in the whole area."""
        g = torch.Generator(device="cpu").manual_seed(
            (seed if seed is not None else self.sc.seed) + 1337)
        t = torch.linspace(0.0, 1.0, self.sc.N + 2)[1:-1]              # (N,)
        line = self.start.cpu() + (self.end.cpu() - self.start.cpu()) * t.unsqueeze(-1)  # (N,2)
        base = line.view(1, -1).repeat(S, 1)                            # (S, 2N)
        noise = torch.randn(S, self.dim, generator=g) * noise_frac * self.sc.area
        return (base + noise).clamp(0.0, self.sc.area).to(self.device)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode (..., 2N) flat vectors into (..., N+2, 2) trajectories with
        fixed start/end clamped to the arena."""
        B = x.shape[:-1]
        q = x.view(*B, self.sc.N, 2).clamp(0.0, self.sc.area)
        start = self.start.expand(*B, 1, 2)
        end = self.end.expand(*B, 1, 2)
        return torch.cat([start, q, end], dim=-2)         # (..., N+2, 2)

    # ---------- fitness ----------
    def fitness(self, x: torch.Tensor) -> torch.Tensor:
        """Lower-is-better cost. x: (..., 2N)."""
        traj = self.decode(x)                              # (..., N+2, 2)
        e = trajectory_energy(traj, self.sc.dt, self.prop) # (..., )

        seg = traj[..., 1:, :] - traj[..., :-1, :]
        v = seg.norm(dim=-1) / self.sc.dt                  # (..., N+1)
        v_over = torch.clamp(v - self.sc.V_max, min=0.0)
        pen_v = self.sc.w_vel * (v_over.pow(2).sum(dim=-1))

        # rate at each slot toward each node
        # uav positions during each slot taken as the slot endpoint
        uav_xy = traj[..., 1:, :]                          # (..., N+1, 2)
        # broadcasting: rate shape (..., N+1, K)
        rate = achievable_rate_bps(uav_xy, self.sc.H, self.nodes, self.chan)
        # service-radius gating: only nodes within horizontal range serve
        horiz = (uav_xy.unsqueeze(-2) - self.nodes).norm(dim=-1)
        rate = rate * (horiz <= self.sc.service_radius_m).to(rate.dtype)
        # nearest-node association: each slot picks the node with max rate
        best_rate, best_idx = rate.max(dim=-1)             # (..., N+1)
        # bits collected per slot: rate * dt -> scatter into per-node bin
        bits_slot = best_rate * self.sc.dt / 1e6           # Mbits per slot
        # accumulate into nodes
        K = self.sc.K
        flat_idx = best_idx                                # (..., N+1)
        bits_per_node = torch.zeros(*x.shape[:-1], K, device=self.device, dtype=bits_slot.dtype)
        bits_per_node.scatter_add_(-1, flat_idx, bits_slot)

        shortfall = torch.clamp(self.data_required - bits_per_node, min=0.0)
        pen_d = self.sc.w_data * shortfall.sum(dim=-1)

        return e + pen_v + pen_d

    def metrics(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Detailed metrics for reporting (no penalties baked in)."""
        traj = self.decode(x)
        e = trajectory_energy(traj, self.sc.dt, self.prop)
        seg = traj[..., 1:, :] - traj[..., :-1, :]
        v = seg.norm(dim=-1) / self.sc.dt
        v_over = torch.clamp(v - self.sc.V_max, min=0.0).sum(dim=-1)

        uav_xy = traj[..., 1:, :]
        rate = achievable_rate_bps(uav_xy, self.sc.H, self.nodes, self.chan)
        horiz = (uav_xy.unsqueeze(-2) - self.nodes).norm(dim=-1)
        rate = rate * (horiz <= self.sc.service_radius_m).to(rate.dtype)
        best_rate, best_idx = rate.max(dim=-1)
        bits_slot = best_rate * self.sc.dt / 1e6
        K = self.sc.K
        bits_per_node = torch.zeros(*x.shape[:-1], K, device=self.device, dtype=bits_slot.dtype)
        bits_per_node.scatter_add_(-1, best_idx, bits_slot)
        delivered = torch.minimum(bits_per_node, self.data_required).sum(dim=-1)
        required = self.data_required.sum()
        completion = delivered / required
        path_len = seg.norm(dim=-1).sum(dim=-1)
        return {
            "energy_j": e,
            "completion": completion,
            "path_length_m": path_len,
            "velocity_violation": v_over,
            "trajectory": traj,
        }
