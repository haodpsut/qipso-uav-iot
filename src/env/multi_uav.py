"""Multi-UAV extension of the UAV-IoT environment.

We split the K ground nodes into M disjoint clusters (K-means by default),
assign one cluster to each UAV, and optimize the M trajectories jointly.
Decision vector has dimension M * N * 2; fitness is the SUM of per-UAV
costs plus a soft collision penalty when two UAVs come within `safe_dist`
of each other at the same time slot.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .channel import ChannelParams, achievable_rate_bps
from .energy_model import PropulsionParams, trajectory_energy
from .uav_iot_env import Scenario


def kmeans_assign(points: torch.Tensor, M: int, iters: int = 50,
                  seed: int = 0) -> torch.Tensor:
    """Simple K-means for cluster assignment (M centers). Returns labels in
    [0..M-1] of shape (K,)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    K = points.shape[0]
    idx0 = torch.randperm(K, generator=gen)[:M]
    centers = points[idx0].clone()
    labels = torch.zeros(K, dtype=torch.long, device=points.device)
    for _ in range(iters):
        d = (points.unsqueeze(1) - centers.unsqueeze(0)).norm(dim=-1)
        new_labels = d.argmin(dim=1)
        if torch.equal(new_labels, labels):
            break
        labels = new_labels
        for m in range(M):
            mask = labels == m
            if mask.any():
                centers[m] = points[mask].mean(dim=0)
    return labels


@dataclass
class MultiUAVScenario:
    base: Scenario
    M: int = 2                              # number of UAVs
    starts: list = None                     # [(x, y), ...] or None -> all base.start
    ends: list = None                       # idem
    safe_dist: float = 30.0
    w_collide: float = 50.0                 # J per m^2 of overlap


class MultiUAVEnv:
    def __init__(self, ms: MultiUAVScenario,
                 prop: PropulsionParams | None = None,
                 chan: ChannelParams | None = None):
        self.ms = ms
        sc = ms.base
        self.sc = sc
        self.prop = prop or PropulsionParams()
        self.chan = chan or ChannelParams()
        self.device = torch.device(sc.device)

        g = torch.Generator(device="cpu").manual_seed(sc.seed)
        nodes = torch.rand(sc.K, 2, generator=g) * sc.area
        d_lo, d_hi = sc.data_per_node_mbits
        data = torch.rand(sc.K, generator=g) * (d_hi - d_lo) + d_lo
        self.nodes = nodes.to(self.device)
        self.data_required = data.to(self.device)
        self.labels = kmeans_assign(self.nodes, ms.M, seed=sc.seed).to(self.device)

        starts = ms.starts or [sc.start] * ms.M
        ends = ms.ends or [sc.end] * ms.M
        self.starts = torch.tensor(starts, device=self.device)
        self.ends = torch.tensor(ends, device=self.device)
        self.dim = ms.M * sc.N * 2

    def warm_start(self, S: int, noise_frac: float = 0.01,
                   seed: int | None = None) -> torch.Tensor:
        """Per-UAV straight-line warm start (start[m] -> end[m]) plus noise."""
        g = torch.Generator(device="cpu").manual_seed(
            (seed if seed is not None else self.sc.seed) + 1337)
        t = torch.linspace(0.0, 1.0, self.sc.N + 2)[1:-1]              # (N,)
        # (M, N, 2) interpolated per-UAV
        st = self.starts.cpu()
        en = self.ends.cpu()
        lines = st.unsqueeze(1) + (en - st).unsqueeze(1) * t.view(1, -1, 1)
        base = lines.reshape(1, -1).repeat(S, 1)                        # (S, M*N*2)
        noise = torch.randn(S, self.dim, generator=g) * noise_frac * self.sc.area
        return (base + noise).clamp(0.0, self.sc.area).to(self.device)

    def _decode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., M*N*2) -> (..., M, N+2, 2)"""
        B = x.shape[:-1]
        q = x.view(*B, self.ms.M, self.sc.N, 2).clamp(0.0, self.sc.area)
        view_shape = (1,) * len(B) + (self.ms.M, 1, 2)
        s = self.starts.view(*view_shape).expand(*B, self.ms.M, 1, 2)
        e = self.ends.view(*view_shape).expand(*B, self.ms.M, 1, 2)
        return torch.cat([s, q, e], dim=-2)

    def fitness(self, x: torch.Tensor) -> torch.Tensor:
        traj = self._decode(x)                                # (..., M, N+2, 2)
        # per-UAV propulsion energy
        e_per = trajectory_energy(traj, self.sc.dt, self.prop)  # (..., M)
        e = e_per.sum(dim=-1)

        # velocity violation
        seg = traj[..., 1:, :] - traj[..., :-1, :]
        v = seg.norm(dim=-1) / self.sc.dt
        v_over = torch.clamp(v - self.sc.V_max, min=0.0)
        pen_v = self.sc.w_vel * v_over.pow(2).sum(dim=(-1, -2))

        # collision penalty: any pair of UAVs at same slot within safe_dist
        pos = traj[..., 1:, :]                                # (..., M, N+1, 2)
        if self.ms.M >= 2:
            d = pos.unsqueeze(-3) - pos.unsqueeze(-4)         # (..., M, M, N+1, 2)
            d = d.norm(dim=-1)
            # zero out the diagonal
            eye = torch.eye(self.ms.M, dtype=torch.bool, device=self.device)
            d = d.masked_fill(eye.unsqueeze(-1), float("inf"))
            overlap = torch.clamp(self.ms.safe_dist - d, min=0.0)
            pen_c = self.ms.w_collide * overlap.pow(2).sum(dim=(-1, -2, -3)) * 0.5
        else:
            pen_c = torch.zeros_like(e)

        # data collection: each UAV only collects from its cluster
        uav_xy = traj[..., 1:, :]                              # (..., M, N+1, 2)
        rate = achievable_rate_bps(uav_xy, self.sc.H,
                                   self.nodes, self.chan)      # (..., M, N+1, K)
        # service-radius gating
        horiz = (uav_xy.unsqueeze(-2) - self.nodes).norm(dim=-1)
        rate = rate * (horiz <= self.sc.service_radius_m).to(rate.dtype)
        # mask out nodes not assigned to UAV m
        cluster_mask = (self.labels.unsqueeze(0) ==
                        torch.arange(self.ms.M, device=self.device).unsqueeze(-1))  # (M, K)
        mask_shape = (1,) * (rate.dim() - 3) + (self.ms.M, 1, self.sc.K)
        rate = rate * cluster_mask.view(*mask_shape).to(rate.dtype)
        best_rate, best_idx = rate.max(dim=-1)                # (..., M, N+1)
        bits_slot = best_rate * self.sc.dt / 1e6
        K = self.sc.K
        bits_per_node = torch.zeros(*x.shape[:-1], self.ms.M, K,
                                    device=self.device, dtype=bits_slot.dtype)
        bits_per_node.scatter_add_(-1, best_idx, bits_slot)
        total_bits = bits_per_node.sum(dim=-2)                # (..., K)
        shortfall = torch.clamp(self.data_required - total_bits, min=0.0)
        pen_d = self.sc.w_data * shortfall.sum(dim=-1)

        return e + pen_v + pen_d + pen_c

    def metrics(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        traj = self._decode(x)
        e_per = trajectory_energy(traj, self.sc.dt, self.prop)
        e_total = e_per.sum(dim=-1)
        uav_xy = traj[..., 1:, :]
        rate = achievable_rate_bps(uav_xy, self.sc.H,
                                   self.nodes, self.chan)
        horiz = (uav_xy.unsqueeze(-2) - self.nodes).norm(dim=-1)
        rate = rate * (horiz <= self.sc.service_radius_m).to(rate.dtype)
        cluster_mask = (self.labels.unsqueeze(0) ==
                        torch.arange(self.ms.M, device=self.device).unsqueeze(-1))
        mask_shape = (1,) * (rate.dim() - 3) + (self.ms.M, 1, self.sc.K)
        rate = rate * cluster_mask.view(*mask_shape).to(rate.dtype)
        best_rate, best_idx = rate.max(dim=-1)
        bits_slot = best_rate * self.sc.dt / 1e6
        bits_per_node = torch.zeros(*x.shape[:-1], self.ms.M, self.sc.K,
                                    device=self.device, dtype=bits_slot.dtype)
        bits_per_node.scatter_add_(-1, best_idx, bits_slot)
        delivered = torch.minimum(bits_per_node.sum(dim=-2),
                                  self.data_required).sum(dim=-1)
        required = self.data_required.sum()
        return {
            "energy_j": e_total,
            "energy_per_uav_j": e_per,
            "completion": delivered / required,
            "trajectory": traj,
            "labels": self.labels,
        }
