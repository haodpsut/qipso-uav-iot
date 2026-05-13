"""Air-to-ground (A2G) channel model for UAV-IoT link.

Uses log-distance path loss with LoS / NLoS probability following the
classical ITU/3GPP-style A2G channel (Al-Hourani 2014). Returns achievable
rate (bps) per slot under a fixed transmit power and bandwidth.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ChannelParams:
    fc: float = 2.4e9            # carrier frequency (Hz)
    B: float = 1e6               # bandwidth (Hz)
    Pt_dbm: float = 23.0         # IoT node transmit power (dBm) -> uplink
    noise_dbm_hz: float = -174.0 # thermal noise PSD (dBm/Hz)
    nf_db: float = 7.0           # receiver noise figure (dB)
    a_los: float = 9.61          # Al-Hourani urban env. param
    b_los: float = 0.16
    eta_los: float = 1.0         # extra LoS attenuation (dB)
    eta_nlos: float = 20.0       # extra NLoS attenuation (dB)


def los_probability(elevation_deg: torch.Tensor, p: ChannelParams) -> torch.Tensor:
    return 1.0 / (1.0 + p.a_los * torch.exp(-p.b_los * (elevation_deg - p.a_los)))


def path_loss_db(uav_xy: torch.Tensor, uav_h: float, node_xy: torch.Tensor,
                 p: ChannelParams | None = None) -> torch.Tensor:
    """Mean path loss (dB) between UAV at (xy, h) and ground node(s).

    uav_xy: (..., 2) ; node_xy: (..., K, 2). Broadcasts on leading dims.
    """
    p = p or ChannelParams()
    # Always inject a singleton "node" dim so we get one rate per (uav-pos, node).
    # uav_xy : (..., 2)    -> (..., 1, 2)
    # node_xy: (..., K, 2)  (or (K, 2))  -> broadcasts on the new -2 axis
    uav_xy = uav_xy.unsqueeze(-2)
    horiz = (uav_xy - node_xy).norm(dim=-1)
    dist = torch.sqrt(horiz.pow(2) + uav_h ** 2)
    elev_deg = torch.atan2(torch.full_like(horiz, uav_h), horiz) * 180.0 / torch.pi
    p_los = los_probability(elev_deg, p)
    c = 3e8
    fspl = 20.0 * torch.log10(torch.clamp(dist, min=1.0)) + \
           20.0 * torch.log10(torch.tensor(p.fc, dtype=dist.dtype, device=dist.device)) + \
           20.0 * torch.log10(torch.tensor(4.0 * torch.pi / c, dtype=dist.dtype, device=dist.device))
    return fspl + p_los * p.eta_los + (1.0 - p_los) * p.eta_nlos


def achievable_rate_bps(uav_xy: torch.Tensor, uav_h: float, node_xy: torch.Tensor,
                        p: ChannelParams | None = None) -> torch.Tensor:
    """Shannon rate (bps) at each (uav, node) pair."""
    p = p or ChannelParams()
    pl_db = path_loss_db(uav_xy, uav_h, node_xy, p)
    # received power (dBm) = Pt - PL
    noise_dbm = p.noise_dbm_hz + 10.0 * torch.log10(torch.tensor(p.B)) + p.nf_db
    snr_db = p.Pt_dbm - pl_db - noise_dbm
    snr_lin = torch.pow(10.0, snr_db / 10.0)
    return p.B * torch.log2(1.0 + snr_lin)
