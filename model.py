import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg
from data import build_knn_from_coords, build_knn_idx_and_relpos

# -------------------------
# TGating helpers
# -------------------------
def _robust_z(d, eps=1e-6):
    med = d.median(dim=1, keepdim=True).values
    mad = (d - med).abs().median(dim=1, keepdim=True).values
    return (d - med) / (mad*1.4826 + eps)

def _residual_frame(x):
    base = x.clone()
    base[:,2:] = (x[:,1:-1] + x[:,0:-2] + x[:,2:]) / 3.0
    base[:,1]  = x[:,0]
    return x - base

def _gating_score(x_raw, lags=(1,2,4,6), alpha=0.6):
    x_res = _residual_frame(x_raw)
    diffs = []
    for ell in lags:
        pad = x_res[:, :ell]
        shifted = torch.cat([pad, x_res[:, :-ell]], 1)
        d = (x_res - shifted).abs().mean(dim=(2,3,4))
        diffs.append(F.softplus(_robust_z(d)))
    g = torch.stack(diffs, -1).mean(-1)
    out=[]
    for b in range(g.size(0)):
        ema=g[b,0]; buf=[ema]
        for t in range(1,g.size(1)):
            ema=alpha*ema+(1-alpha)*g[b,t]; buf.append(ema)
        out.append(torch.stack(buf))
    g=torch.stack(out,0)
    return torch.sigmoid(1.5*(g - g.mean(dim=1, keepdim=True))).detach()

def compute_gating_signal(x_raw):
    return _gating_score(x_raw)

# -------------------------
# Embedding + SFE
# -------------------------
class MultiPeriodicEmbed(nn.Module):
    def __init__(self, d_model, periods=[24, 72]):
        super().__init__()
        self.periods = periods
        n = len(periods)
        base = d_model // n
        rem  = d_model - base * n
        self._parts = [base + (1 if i < rem else 0) for i in range(n)]
        self.period_embeds = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2, p),
                nn.GELU(),
                nn.Linear(p, p)
            ) for p in self._parts
        ])
    def forward(self, time_idx):
        embeds = []
        for (period, layer) in zip(self.periods, self.period_embeds):
            phase = 2 * math.pi * time_idx / period
            sin_cos = torch.stack([torch.sin(phase), torch.cos(phase)], dim=-1)
            embeds.append(layer(sin_cos))
        return torch.cat(embeds, dim=-1)

class SFE(nn.Module):
    def __init__(self, d_model, num_frequencies=16):
        super().__init__()
        self.num_freq = num_frequencies
        self.sfe_processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model)
            ) for _ in range(num_frequencies)
        ])
        self.combiner = nn.Sequential(
            nn.Linear(d_model * num_frequencies, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
    def forward(self, x):
        B, P, N, d = x.shape
        x_reshaped = x.permute(0, 2, 3, 1)
        x_freq = torch.fft.rfft(x_reshaped, dim=-1)
        freq_bins = x_freq.shape[-1]
        processed_freqs = []
        for i in range(min(self.num_freq, freq_bins)):
            real_comp = x_freq[:,:,:,i].real
            imag_comp = x_freq[:,:,:,i].imag
            comp = torch.cat([real_comp, imag_comp], dim=-1)
            processed = self.sfe_processors[i](comp)
            processed_freqs.append(processed)
        if len(processed_freqs) == 0:
            processed_freqs = [torch.zeros(B, N, d, device=x.device)]
        while len(processed_freqs) < self.num_freq:
            processed_freqs.append(torch.zeros_like(processed_freqs[0]))
        combined = torch.cat(processed_freqs, dim=-1)
        output = self.combiner(combined)
        return output.unsqueeze(1).expand(-1, P, -1, -1)

# -------------------------
# Temporal module
# -------------------------
class TGating(nn.Module):
    def __init__(self, C_in, d_model, P):
        super().__init__()
        self.P = P
        self.d_model = d_model
        self.in_proj = nn.Linear(C_in, d_model)
        self.periodic_embed = MultiPeriodicEmbed(d_model//2, cfg.PERIODS)
        self.noise_embed = nn.Linear(C_in, d_model//2)
        self.sfe = SFE(d_model, cfg.NUM_FREQUENCIES)
        self.pool_param = nn.Parameter(torch.randn(P, 1))

    def forward(self, x, gating_hint=None):
        B, P, H, W, C = x.shape
        N = H * W
        x_flat = x.reshape(B, P, N, C)
        h = self.in_proj(x_flat)

        time_idx = torch.arange(P, device=x.device).float().unsqueeze(0).expand(B, -1)
        periodic_emb = self.periodic_embed(time_idx)

        spatial_mean = x_flat.mean(dim=2)
        noise_emb = self.noise_embed(spatial_mean)

        time_emb = torch.cat([periodic_emb, noise_emb], dim=-1)
        h = h + time_emb.unsqueeze(2)

        if cfg.USE_SFE:
            h_freq = self.sfe(h)
            h = h + 0.3 * h_freq

        base_att = torch.softmax(self.pool_param.squeeze(-1), dim=0)

        if cfg.USE_TGATING:
            if gating_hint is None:
                gating = compute_gating_signal(x)
            else:
                gating = gating_hint
                if gating.dim() == 1:
                    gating = gating.unsqueeze(0).expand(B, -1)
            att = base_att[None, :] * (1.0 + cfg.TGATING_GAIN * gating)
            att = torch.softmax(att, dim=1)
        else:
            att = base_att[None, :].expand(B, -1)

        att = att[:, :, None, None]
        z = (h * att).sum(dim=1)
        time_vec = time_emb.mean(dim=1)
        return z, time_vec

# -------------------------
# Spatial modules: GOT + MGAT
# -------------------------
class GOT(nn.Module):
    def __init__(self, d_model, H, W, K, eps=0.1, use_emb_dist=True, tau=1.0, dropout=0.1, coords: torch.Tensor = None):
        super().__init__()
        self.K = K
        self.proto = nn.Parameter(torch.randn(K, d_model) * 0.02)
        self.proj_q = nn.Linear(d_model, d_model, bias=False)
        self.proj_p = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(),
                                 nn.Linear(4*d_model, d_model), nn.Dropout(dropout))
        self.eps = eps
        self.tau = tau
        self.use_emb_dist = use_emb_dist

        if coords is None:
            yy, xx = torch.meshgrid(torch.linspace(-1,1,H), torch.linspace(-1,1,W), indexing="ij")
            coords_b = torch.stack([yy, xx], dim=-1).reshape(-1,2)
        else:
            coords_b = coords.float()
            if coords_b.dim() != 2:
                raise ValueError(f"coords must be (N,2), got {coords_b.shape}")
            if coords_b.size(1) != 2:
                coords_b = coords_b[:, :2].contiguous()

        self.register_buffer("coords", coords_b, persistent=False)
        self.proto_xy = nn.Parameter(torch.randn(K, 2) * 0.1)

    def forward(self, z, margin: torch.Tensor = None, eps_override: float = None, tau_override: float = None):
        B, N, D = z.shape
        device = z.device
        coords = self.coords.to(device)

        q = self.proj_q(z)
        p = self.proj_p(self.proto).unsqueeze(0).expand(B,-1,-1)
        proto_xy = torch.tanh(self.proto_xy)

        geod = torch.cdist(coords.unsqueeze(0), proto_xy.unsqueeze(0), p=2.0).expand(B, -1, -1)

        if self.use_emb_dist:
            qn = F.normalize(q, dim=-1); pn = F.normalize(p, dim=-1)
            emb = 1.0 - torch.einsum("bnd,bkd->bnk", qn, pn)
            C = geod + emb
        else:
            C = geod

        eps = float(self.eps if eps_override is None else eps_override)
        tau = float(self.tau if tau_override is None else tau_override)

        logits = -C / (eps + 1e-6)
        if margin is not None:
            logits = logits + margin

        Tm = torch.softmax(tau * logits, dim=-1)
        mixed = torch.einsum("bnk,bkd->bnd", Tm, p)

        out = self.norm(z + mixed)
        out = self.norm(out + self.ffn(out))
        return out, Tm, geod

class MGAT(nn.Module):
    def __init__(self, d_model, H, W, K=16, tv_w=1e-3, geo_w=0.25, sim_w=1.0, dropout=0.1, coords: torch.Tensor = None):
        super().__init__()
        self.H, self.W, self.K = H, W, K
        self.tv_w = tv_w
        self.geo_w = geo_w
        self.sim_w = sim_w

        self.to_q = nn.Linear(d_model, d_model)
        self.to_k = nn.Linear(d_model, d_model)
        self.to_v = nn.Linear(d_model, d_model)
        self.out  = nn.Sequential(nn.Linear(d_model, d_model), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(d_model)
        self.dk = d_model

        self.s_summary = nn.Sequential(nn.Linear(2, 16), nn.GELU(), nn.Linear(16, 16), nn.GELU())
        self.z_gate = nn.Sequential(nn.Linear(d_model, 16), nn.GELU())
        self.param_head = nn.Linear(32, 3)
        self.scale_gate = nn.Linear(32, 3)

        self._coords_user = coords
        self.register_buffer("_knn_idx", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("_relpos", torch.empty(0), persistent=False)
        self.register_buffer("_geod", torch.empty(0), persistent=False)

        self.drop_path = nn.Dropout(p=0.1)

    def _ensure_graph(self, device, N_expected: int):
        if self._knn_idx.numel() == 0 or self._knn_idx.device != device:
            if self._coords_user is not None:
                coords = self._coords_user.to(device).float()
                if coords.dim() != 2:
                    raise ValueError(f"coords must be (N,2), got {coords.shape}")
                if coords.size(1) != 2:
                    coords = coords[:, :2].contiguous()
                if coords.size(0) != N_expected:
                    raise ValueError(f"coords N mismatch: coords has N={coords.size(0)} but z has N={N_expected}")
                idx, rel, geo = build_knn_from_coords(coords, cfg.DLAK_K)
                rel = rel / (rel.abs().amax(dim=(0,1), keepdim=True).clamp_min(1e-6))
            else:
                idx, rel, geo, _ = build_knn_idx_and_relpos(self.H, self.W, cfg.DLAK_K, device)
            self._knn_idx = idx
            self._relpos = rel
            self._geod = geo

    def _gather_neighbors(self, x, idx):
        B = x.shape[0]
        b = torch.arange(B, device=idx.device)[:, None, None]
        n_idx = idx[None, :, :]
        return x[b, n_idx]

    def _tv2d(self, tensor_2d):
        dy = (tensor_2d[:,1:,:] - tensor_2d[:,:-1,:]).abs().mean()
        dx = (tensor_2d[:,:,1:] - tensor_2d[:,:,:-1]).abs().mean()
        return dx + dy

    def forward(self, z, S):
        device = z.device
        B,N,D = z.shape
        self._ensure_graph(device, N_expected=N)

        idx = self._knn_idx; rel = self._relpos; geod = self._geod

        q = self.to_q(z); k = self.to_k(z); v = self.to_v(z)
        kN = self._gather_neighbors(k, idx)
        vN = self._gather_neighbors(v, idx)
        S_n = self._gather_neighbors(S, idx)

        m1 = S.mean(dim=-1, keepdim=True)
        m2 = (S*S).mean(dim=-1, keepdim=True)
        s_feat = torch.cat([m1, m2], dim=-1)
        s_emb  = self.s_summary(s_feat)
        z_emb  = self.z_gate(z)
        ker_in = torch.cat([s_emb, z_emb], dim=-1)

        a,b,c  = self.param_head(ker_in).unbind(-1)
        La = torch.exp(a).unsqueeze(-1).unsqueeze(-1)
        Lb = b.unsqueeze(-1).unsqueeze(-1)
        Lc = torch.exp(c).unsqueeze(-1).unsqueeze(-1)
        pad = torch.zeros_like(La)
        L = torch.cat([torch.cat([La, pad], dim=-1),
                       torch.cat([Lb, Lc], dim=-1)], dim=-2)
        A = torch.matmul(L, L.transpose(-1,-2))

        rel_b = rel.unsqueeze(0).expand(B, -1, -1, -1)
        rAr   = torch.einsum('bnkd,bndc,bnkc->bnk', rel_b, A, rel_b)

        S_q = S.unsqueeze(2)
        sim = (S_q * S_n).sum(-1)

        content = (q.unsqueeze(2) * kN).sum(-1) / math.sqrt(self.dk)

        SC = [dict(r=1.5, s0=0.25, sl=0.20, cb=0.70),
              dict(r=2.5, s0=0.35, sl=0.20, cb=0.50),
              dict(r=4.0, s0=0.55, sl=0.25, cb=0.30)]

        logits_scales = []
        geod_b = geod.unsqueeze(0)
        big_neg = -1e4

        for sc in SC:
            hard_mask = (geod > sc["r"]).to(z.dtype).unsqueeze(0)
            sigma = (sc["s0"] + sc["sl"]*geod).unsqueeze(0)
            rAr_scaled = rAr / (sigma*sigma).clamp_min(1e-6)

            center_prior = torch.exp(-(geod_b**2) / (cfg.MGAT_R0**2))
            center_term  = sc["cb"] * center_prior

            att_i = content - rAr_scaled - 0.25*geod_b + 1.0*sim + center_term
            att_i = att_i + hard_mask*big_neg
            logits_scales.append(att_i)

        g = torch.softmax(self.scale_gate(ker_in), dim=-1)
        att_logits = (logits_scales[0] * g[...,0:1] +
                      logits_scales[1] * g[...,1:2] +
                      logits_scales[2] * g[...,2:3])

        att = torch.softmax(att_logits, dim=-1)
        out = (att.unsqueeze(-1) * vN).sum(2)
        out = self.norm(z + self.drop_path(self.out(out)))

        if cfg.NODE_MODE:
            reg = torch.tensor(0.0, device=z.device)
        else:
            A_norm = A.reshape(B, self.H, self.W, 2, 2).norm(dim=(-1,-2))
            reg = self.tv_w * self._tv2d(A_norm)

        return out, reg, {}

class SpatialEncoder(nn.Module):
    def __init__(self, d, H, W, n_proto, coords: torch.Tensor = None):
        super().__init__()
        self.H, self.W = H, W
        self.got = GOT(
            d_model=d, H=H, W=W, K=n_proto,
            eps=cfg.GOT_EPS,
            use_emb_dist=cfg.GOT_USE_EMB_DIST,
            tau=cfg.GOT_TAU,
            dropout=cfg.DROPOUT,
            coords=coords
        )
        self.mgat = MGAT(
            d_model=d, H=H, W=W, K=cfg.DLAK_K,
            tv_w=1e-3, geo_w=0.25, sim_w=1.0, dropout=cfg.DROPOUT,
            coords=coords
        ) if cfg.USE_MGAT else None

    def forward(self, z_cell):
        if not cfg.USE_GOT:
            return z_cell, torch.tensor(0.0, device=z_cell.device)
        h1, S1, _ = self.got(z_cell)
        if self.mgat is not None:
            h_mgat1, reg1, _ = self.mgat(h1, S1)
            return h_mgat1, reg1
        return h1, torch.tensor(0.0, device=z_cell.device)

class SpecTraNet(nn.Module):
    def __init__(self, H, W, C, P, coords: torch.Tensor = None):
        super().__init__()
        self.temporal = TGating(C, cfg.D_MODEL, P)
        self.spatial  = SpatialEncoder(cfg.D_MODEL, H, W, cfg.N_PROTO, coords=coords)
        self.out_proj = nn.Linear(cfg.D_MODEL, C)
        self.H=H; self.W=W

    def forward(self, x, gating_hint=None):
        B = x.size(0)
        last_in = x[:, -1]
        z_cell, _ = self.temporal(x, gating_hint=gating_hint)
        h, reg_spa = self.spatial(z_cell)
        y_delta = self.out_proj(h).view(B, self.H, self.W, -1)
        y = last_in + y_delta if cfg.USE_RESIDUAL_TO_LAST else y_delta
        return y, reg_spa

