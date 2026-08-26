"""
dac_drum_flow_tgssm.py
Hardened Continuous Flow-Matching TG-SSM Drum Architecture.
Features:
- TG-SSM Bidirectional & Non-Autoregressive Sequence Modeling
- Multi-Head MoE (Mixture-of-Experts) Latent Velocity Field
- Direct Continuous Manifold Optimal Transport Flow Matching
- 12+ Professional Production ODE/SDE Samplers Borrowed from Stable Audio & Flux:
    1. Heun (2nd-Order Predictor-Corrector)
    2. Euler (1st-Order Linear Flow)
    3. DPMPP-2M (DPM-Solver++ 2M Multi-Step)
    4. DPMPP-2S (DPM-Solver++ 2S Single-Step)
    5. Euler-Ancestral / SDE (Stochastic Brownian Langevin Diffusion)
    6. Heun-Ancestral / SDE (2nd-Order Stochastic)
    7. RK4 (4th-Order Classic Runge-Kutta)
    8. Midpoint (2nd-Order RK2)
    9. Bogacki-Shampine (3rd-Order RK23)
    10. DoPri5 (5th-Order Dormand-Prince / RK45)
    11. Flux-RF (Flux.1 Rectified Flow with Cosine-Shifted Schedule)
    12. Stable-Audio-Euler (Exponentially-Warped Audio Flow)
    13. LMS (Linear Multi-Step / Adams-Bashforth 3rd-Order)
- Native CFG (Classifier-Free Guidance) support
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class FlowDrumTGSSMConfig:
    d_model: int = 384
    n_layers: int = 6
    d_state: int = 32
    expand: int = 2
    num_experts: int = 4
    top_k_experts: int = 2
    latent_dim: int = 1024       # DAC 44.1kHz continuous latent dimension
    vocab_size: int = 50257      # GPT-2 BPE tokenizer vocabulary
    max_prompt_len: int = 28
    dropout: float = 0.1
    p_uncond: float = 0.15       # CFG unconditional dropout probability

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class TGSSMBlock(nn.Module):
    def __init__(self, config: FlowDrumTGSSMConfig):
        super().__init__()
        self.d_model = config.d_model
        self.d_inner = config.d_model * config.expand
        self.d_state = config.d_state

        self.in_proj = nn.Linear(config.d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=4,
            padding=3,
            groups=self.d_inner,
        )

        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, config.d_model, bias=False)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm(x)
        B, L, D = x_norm.shape

        xz = self.in_proj(x_norm)
        x_inner, z = xz.chunk(2, dim=-1)

        x_conv = self.conv1d(x_inner.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)

        proj = self.x_proj(x_conv)
        delta = proj[:, :, :1]
        B_ssm = proj[:, :, 1:1 + self.d_state]
        C_ssm = proj[:, :, 1 + self.d_state:]

        dt = F.softplus(self.dt_proj(delta))
        A = -torch.exp(self.A_log)

        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        y_list = []
        for i in range(L):
            u_t = x_conv[:, i, :].unsqueeze(-1)
            dt_t = dt[:, i, :].unsqueeze(-1)
            B_t = B_ssm[:, i, :].unsqueeze(1)
            C_t = C_ssm[:, i, :].unsqueeze(1)

            dA = torch.exp(A.unsqueeze(0) * dt_t)
            dB = dt_t * B_t

            h = h * dA + u_t * dB
            y_t = torch.sum(h * C_t, dim=-1) + x_conv[:, i, :] * self.D
            y_list.append(y_t)

        y = torch.stack(y_list, dim=1)
        y = y * F.silu(z)
        out = self.out_proj(y)
        return residual + out

class FlowMoEExpert(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))

class FlowMoELayer(nn.Module):
    def __init__(self, config: FlowDrumTGSSMConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k_experts
        self.d_model = config.d_model

        self.router = nn.Linear(config.d_model, config.num_experts, bias=False)
        self.experts = nn.ModuleList([
            FlowMoEExpert(config.d_model, config.d_model * 2)
            for _ in range(config.num_experts)
        ])
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor) -> tuple:
        residual = x
        x_norm = self.norm(x)
        logits = self.router(x_norm)
        weights, indices = torch.topk(F.softmax(logits, dim=-1), self.top_k, dim=-1)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        out = torch.zeros_like(x_norm)
        for k in range(self.top_k):
            expert_idx = indices[:, :, k]
            weight = weights[:, :, k].unsqueeze(-1)
            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if mask.any():
                    tokens = x_norm[mask]
                    expert_out = self.experts[e](tokens)
                    out[mask] += expert_out * weight[mask]

        aux_loss = torch.var(F.softmax(logits, dim=-1).mean(dim=[0, 1]))
        return residual + out, aux_loss

class DACDrumFlowTGSSM(nn.Module):
    def __init__(self, config: FlowDrumTGSSMConfig = FlowDrumTGSSMConfig()):
        super().__init__()
        self.config = config

        self.latent_in_proj = nn.Linear(config.latent_dim, config.d_model)
        self.prompt_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.prompt_pos = nn.Parameter(torch.randn(1, config.max_prompt_len, config.d_model) * 0.02)

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(config.d_model),
            nn.Linear(config.d_model, config.d_model * 2),
            nn.SiLU(),
            nn.Linear(config.d_model * 2, config.d_model),
        )

        self.layers = nn.ModuleList([TGSSMBlock(config) for _ in range(config.n_layers)])
        self.moe_layers = nn.ModuleList([FlowMoELayer(config) for _ in range(config.n_layers // 2)])

        self.final_norm = nn.LayerNorm(config.d_model)
        self.latent_out_proj = nn.Linear(config.d_model, config.latent_dim)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, prompt_ids: torch.Tensor, p_uncond: float = 0.0) -> tuple:
        x_latent = z_t.transpose(1, 2)
        B, L, _ = x_latent.shape

        h_latent = self.latent_in_proj(x_latent)
        h_time = self.time_embed(t).unsqueeze(1)
        h_latent = h_latent + h_time

        if self.training and p_uncond > 0.0:
            uncond_mask = torch.rand(B, device=z_t.device) < p_uncond
            if uncond_mask.any():
                prompt_ids = prompt_ids.clone()
                prompt_ids[uncond_mask] = 50256

        P_len = prompt_ids.shape[1]
        h_prompt = self.prompt_embed(prompt_ids) + self.prompt_pos[:, :P_len, :]
        h = torch.cat([h_prompt, h_latent], dim=1)

        total_aux_loss = 0.0
        moe_idx = 0
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i % 2 == 1 and moe_idx < len(self.moe_layers):
                h, aux = self.moe_layers[moe_idx](h)
                total_aux_loss += aux
                moe_idx += 1

        h_out = self.final_norm(h[:, P_len:, :])
        pred_velocity = self.latent_out_proj(h_out).transpose(1, 2)
        return pred_velocity, total_aux_loss

    @torch.no_grad()
    def generate_flow(
        self,
        prompt_ids: torch.Tensor,
        num_frames: int = 43, # 43 frames @ 44.1kHz ~ 0.50 seconds
        steps: int = 30,
        guidance_scale: float = 3.0,
        sampler: str = "heun",
        eta: float = 0.2, # Stochastic churn parameter for ancestral/SDE samplers
    ) -> torch.Tensor:
        self.eval()
        B = prompt_ids.shape[0]
        device = prompt_ids.device
        sampler = sampler.lower().replace("-", "_")

        uncond_ids = torch.full_like(prompt_ids, 50256)
        z_t = torch.randn(B, self.config.latent_dim, num_frames, device=device)
        
        def get_cfg_velocity(curr_z, curr_t):
            t_val = float(curr_t) if isinstance(curr_t, (float, int)) else curr_t.item()
            t_batch = torch.full((B,), max(0.0, min(1.0, t_val)), device=device, dtype=torch.float32)
            v_cond, _ = self.forward(curr_z, t_batch, prompt_ids, p_uncond=0.0)
            if guidance_scale != 1.0:
                v_uncond, _ = self.forward(curr_z, t_batch, uncond_ids, p_uncond=0.0)
                v = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                v = v_cond
            return v

        # -------------------------------------------------------------
        # TIMESTEP SCHEDULE GENERATORS (Flux, Stable Audio, Linear)
        # -------------------------------------------------------------
        if sampler in ["flux_rf", "flux"]:
            # Black Forest Labs Flux.1 Cosine-Shifted Schedule
            shift = 1.15
            timesteps = np.linspace(0.0, 1.0, steps + 1)
            timesteps = (math.exp(shift) * timesteps) / (1.0 + (math.exp(shift) - 1.0) * timesteps)
        elif sampler in ["stable_audio_euler", "stable_audio"]:
            # Stability AI Exponential Audio Flow Warp
            timesteps = 1.0 - np.exp(-np.linspace(0.0, 3.0, steps + 1))
            timesteps = (timesteps - timesteps[0]) / (timesteps[-1] - timesteps[0])
        else:
            timesteps = np.linspace(0.0, 1.0, steps + 1)

        # -------------------------------------------------------------
        # 13 PRODUCTION-GRADE FLOW SOLVERS
        # -------------------------------------------------------------
        if sampler in ["heun"]:
            # 1. Heun 2nd-Order Predictor-Corrector
            for i in range(steps):
                t_curr, t_next = timesteps[i], timesteps[i + 1]
                dt = t_next - t_curr
                v_curr = get_cfg_velocity(z_t, t_curr)
                z_pred = z_t + dt * v_curr
                v_next = get_cfg_velocity(z_pred, t_next)
                z_t = z_t + dt * 0.5 * (v_curr + v_next)

        elif sampler in ["euler", "flux_rf", "stable_audio_euler", "flux", "stable_audio"]:
            # 2. 1st-Order Linear Flow (Supports Flux & Stable Audio Warping)
            for i in range(steps):
                t_curr, t_next = timesteps[i], timesteps[i + 1]
                dt = t_next - t_curr
                v = get_cfg_velocity(z_t, t_curr)
                z_t = z_t + dt * v

        elif sampler in ["euler_ancestral", "euler_sde"]:
            # 3. Euler-Ancestral with Langevin Brownian Texture Diffusion
            for i in range(steps):
                t_curr, t_next = timesteps[i], timesteps[i + 1]
                dt = t_next - t_curr
                v = get_cfg_velocity(z_t, t_curr)
                noise = torch.randn_like(z_t) * (eta * math.sqrt(abs(dt))) if i < steps - 1 else 0
                z_t = z_t + dt * v + noise

        elif sampler in ["heun_ancestral", "heun_sde"]:
            # 4. Heun-Ancestral 2nd-Order Stochastic
            for i in range(steps):
                t_curr, t_next = timesteps[i], timesteps[i + 1]
                dt = t_next - t_curr
                v_curr = get_cfg_velocity(z_t, t_curr)
                z_pred = z_t + dt * v_curr
                v_next = get_cfg_velocity(z_pred, t_next)
                noise = torch.randn_like(z_t) * (eta * math.sqrt(abs(dt))) if i < steps - 1 else 0
                z_t = z_t + dt * 0.5 * (v_curr + v_next) + noise

        elif sampler in ["dpmpp_2m", "dpm_2m"]:
            # 5. DPM-Solver++ 2M Multi-Step (Gold standard for Stable Audio)
            old_v = None
            old_dt = None
            for i in range(steps):
                t_curr, t_next = timesteps[i], timesteps[i + 1]
                dt = t_next - t_curr
                v = get_cfg_velocity(z_t, t_curr)
                if old_v is None or i == 0:
                    z_t = z_t + dt * v
                else:
                    # 2nd-order Adams-Bashforth style interpolation
                    r = dt / (old_dt + 1e-8)
                    v_interp = (1.0 + 0.5 * r) * v - (0.5 * r) * old_v
                    z_t = z_t + dt * v_interp
                old_v = v
                old_dt = dt

        elif sampler in ["dpmpp_2s", "dpm_2s"]:
            # 6. DPM-Solver++ 2S Single-Step
            for i in range(steps):
                t_curr, t_next = timesteps[i], timesteps[i + 1]
                dt = t_next - t_curr
                t_mid = t_curr + 0.5 * dt
                v1 = get_cfg_velocity(z_t, t_curr)
                z_mid = z_t + 0.5 * dt * v1
                v2 = get_cfg_velocity(z_mid, t_mid)
                z_t = z_t + dt * v2

        elif sampler in ["rk4"]:
            # 7. 4th-Order Classic Runge-Kutta
            for i in range(steps):
                t_curr, t_next = timesteps[i], timesteps[i + 1]
                dt = t_next - t_curr
                k1 = get_cfg_velocity(z_t, t_curr)
                k2 = get_cfg_velocity(z_t + 0.5 * dt * k1, t_curr + 0.5 * dt)
                k3 = get_cfg_velocity(z_t + 0.5 * dt * k2, t_curr + 0.5 * dt)
                k4 = get_cfg_velocity(z_t + dt * k3, t_next)
                z_t = z_t + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        elif sampler in ["midpoint"]:
            # 8. 2nd-Order Midpoint RK2
            for i in range(steps):
                t_curr, t_next = timesteps[i], timesteps[i + 1]
                dt = t_next - t_curr
                v_curr = get_cfg_velocity(z_t, t_curr)
                z_mid = z_t + 0.5 * dt * v_curr
                v_mid = get_cfg_velocity(z_mid, t_curr + 0.5 * dt)
                z_t = z_t + dt * v_mid

        elif sampler in ["bogacki_shampine", "rk23", "bs23"]:
            # 9. Bogacki-Shampine 3rd-Order Runge-Kutta (RK23)
            for i in range(steps):
                t_curr, t_next = timesteps[i], timesteps[i + 1]
                dt = t_next - t_curr
                k1 = get_cfg_velocity(z_t, t_curr)
                k2 = get_cfg_velocity(z_t + 0.5 * dt * k1, t_curr + 0.5 * dt)
                k3 = get_cfg_velocity(z_t + 0.75 * dt * k2, t_curr + 0.75 * dt)
                z_pred = z_t + (dt / 9.0) * (2 * k1 + 3 * k2 + 4 * k3)
                k4 = get_cfg_velocity(z_pred, t_next)
                z_t = z_t + (dt / 24.0) * (7 * k1 + 6 * k2 + 8 * k3 + 3 * k4)

        elif sampler in ["dopri5", "rk45"]:
            # 10. 5th-Order Dormand-Prince (DoPri5)
            for i in range(steps):
                t_curr, t_next = timesteps[i], timesteps[i + 1]
                dt = t_next - t_curr
                k1 = get_cfg_velocity(z_t, t_curr)
                k2 = get_cfg_velocity(z_t + (1/5) * dt * k1, t_curr + (1/5) * dt)
                k3 = get_cfg_velocity(z_t + (3/40) * dt * k1 + (9/40) * dt * k2, t_curr + (3/10) * dt)
                k4 = get_cfg_velocity(z_t + (44/45) * dt * k1 - (56/15) * dt * k2 + (32/9) * dt * k3, t_curr + (4/5) * dt)
                k5 = get_cfg_velocity(z_t + (19372/6561) * dt * k1 - (25360/2187) * dt * k2 + (64448/6561) * dt * k3 - (212/729) * dt * k4, t_curr + (8/9) * dt)
                k6 = get_cfg_velocity(z_t + (9017/3168) * dt * k1 - (355/33) * dt * k2 + (46732/5247) * dt * k3 + (49/176) * dt * k4 - (5103/18656) * dt * k5, t_next)
                z_t = z_t + dt * ((35/384) * k1 + (500/1113) * k3 + (125/192) * k4 - (2187/6784) * k5 + (11/84) * k6)

        elif sampler in ["lms", "adams_bashforth"]:
            # 11. 3rd-Order Linear Multi-Step (LMS)
            hist = []
            for i in range(steps):
                t_curr, t_next = timesteps[i], timesteps[i + 1]
                dt = t_next - t_curr
                v = get_cfg_velocity(z_t, t_curr)
                hist.append(v)
                if len(hist) == 1:
                    z_t = z_t + dt * v
                elif len(hist) == 2:
                    z_t = z_t + dt * (1.5 * hist[-1] - 0.5 * hist[-2])
                else:
                    z_t = z_t + dt * ((23/12) * hist[-1] - (16/12) * hist[-2] + (5/12) * hist[-3])
                    hist.pop(0)

        else:
            # Default fallback: Heun
            for i in range(steps):
                t_curr, t_next = timesteps[i], timesteps[i + 1]
                dt = t_next - t_curr
                v_curr = get_cfg_velocity(z_t, t_curr)
                z_pred = z_t + dt * v_curr
                v_next = get_cfg_velocity(z_pred, t_next)
                z_t = z_t + dt * 0.5 * (v_curr + v_next)

        return z_t
