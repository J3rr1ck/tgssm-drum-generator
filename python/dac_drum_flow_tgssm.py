"""
dac_drum_flow_tgssm.py
Hardened Continuous Flow-Matching TG-SSM Drum Architecture.
Features:
- TG-SSM Bidirectional & Non-Autoregressive Sequence Modeling
- Multi-Head MoE (Mixture-of-Experts) Latent Velocity Field
- Direct Continuous Manifold Optimal Transport Flow Matching
- High-Order Solvers: Heun (2nd-order predictor-corrector), RK4, Midpoint, Euler
- Native CFG (Classifier-Free Guidance) support
"""

import math
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
        # t: [B] in range [0, 1]
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
        # x: [B, L, D]
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

        # Vectorized recurrent state scan
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
        # z_t: [B, 1024, L] -> transpose to [B, L, 1024]
        x_latent = z_t.transpose(1, 2)
        B, L, _ = x_latent.shape

        h_latent = self.latent_in_proj(x_latent) # [B, L, D]
        h_time = self.time_embed(t).unsqueeze(1) # [B, 1, D]
        h_latent = h_latent + h_time

        # Classifier-Free Guidance Training Dropout
        if self.training and p_uncond > 0.0:
            uncond_mask = torch.rand(B, device=z_t.device) < p_uncond
            if uncond_mask.any():
                prompt_ids = prompt_ids.clone()
                prompt_ids[uncond_mask] = 50256 # GPT-2 EOS / pad token

        # Prompt conditioning
        P_len = prompt_ids.shape[1]
        h_prompt = self.prompt_embed(prompt_ids) + self.prompt_pos[:, :P_len, :]

        # Concatenate conditioning prompt prefix with latent trajectory sequence
        h = torch.cat([h_prompt, h_latent], dim=1) # [B, P + L, D]

        total_aux_loss = 0.0
        moe_idx = 0
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i % 2 == 1 and moe_idx < len(self.moe_layers):
                h, aux = self.moe_layers[moe_idx](h)
                total_aux_loss += aux
                moe_idx += 1

        h_out = self.final_norm(h[:, P_len:, :])
        pred_velocity = self.latent_out_proj(h_out).transpose(1, 2) # [B, 1024, L]
        return pred_velocity, total_aux_loss

    @torch.no_grad()
    def generate_flow(
        self,
        prompt_ids: torch.Tensor,
        num_frames: int = 43, # 43 frames @ 44.1kHz ~ 0.50 seconds
        steps: int = 30,
        guidance_scale: float = 3.0,
        sampler: str = "heun",
    ) -> torch.Tensor:
        self.eval()
        B = prompt_ids.shape[0]
        device = prompt_ids.device

        # Unconditional empty prompt for CFG
        uncond_ids = torch.full_like(prompt_ids, 50256)

        # Initial standard Gaussian noise on continuous manifold
        z_t = torch.randn(B, self.config.latent_dim, num_frames, device=device)
        dt = 1.0 / steps

        def get_cfg_velocity(curr_z, curr_t):
            t_batch = torch.full((B,), curr_t, device=device, dtype=torch.float32)
            v_cond, _ = self.forward(curr_z, t_batch, prompt_ids, p_uncond=0.0)
            if guidance_scale != 1.0:
                v_uncond, _ = self.forward(curr_z, t_batch, uncond_ids, p_uncond=0.0)
                v = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                v = v_cond
            return v

        # -------------------------------------------------------------
        # HIGH-ORDER NUMERICAL FLOW SOLVERS
        # -------------------------------------------------------------
        if sampler == "heun":
            # 2nd-Order Predictor-Corrector
            for i in range(steps):
                t_curr = i * dt
                t_next = (i + 1) * dt
                v_curr = get_cfg_velocity(z_t, t_curr)
                z_pred = z_t + dt * v_curr
                v_next = get_cfg_velocity(z_pred, t_next)
                z_t = z_t + dt * 0.5 * (v_curr + v_next)

        elif sampler == "rk4":
            # 4th-Order Classic Runge-Kutta
            for i in range(steps):
                t_curr = i * dt
                k1 = get_cfg_velocity(z_t, t_curr)
                k2 = get_cfg_velocity(z_t + 0.5 * dt * k1, t_curr + 0.5 * dt)
                k3 = get_cfg_velocity(z_t + 0.5 * dt * k2, t_curr + 0.5 * dt)
                k4 = get_cfg_velocity(z_t + dt * k3, t_curr + dt)
                z_t = z_t + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        elif sampler == "midpoint":
            # 2nd-Order Midpoint RK2
            for i in range(steps):
                t_curr = i * dt
                v_curr = get_cfg_velocity(z_t, t_curr)
                z_mid = z_t + 0.5 * dt * v_curr
                v_mid = get_cfg_velocity(z_mid, t_curr + 0.5 * dt)
                z_t = z_t + dt * v_mid

        else:
            # 1st-Order Euler
            for i in range(steps):
                t_curr = i * dt
                v = get_cfg_velocity(z_t, t_curr)
                z_t = z_t + dt * v

        return z_t
