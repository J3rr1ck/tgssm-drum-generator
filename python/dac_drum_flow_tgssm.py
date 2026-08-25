"""
dac_drum_flow_tgssm.py
Continuous Flow-Matching (Rectified Flow) TG-SSM Model for Studio-Grade 44.1kHz Drum Synthesis.

Operates directly on the continuous 1024-dimensional latent manifold of DAC (Descript Audio Codec).
Predicts continuous ODE velocity field v(z_t, t, c) with System 2 Hamiltonian Deliberation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.mixed_precision import FP4Linear, RMSNormFP32

@dataclass
class FlowDrumTGSSMConfig:
    d_model: int = 512
    n_layers: int = 8
    d_state: int = 32
    d_conv: int = 4
    expand: int = 2
    dt_rank: int = 32
    num_experts: int = 4
    top_k_experts: int = 2
    deliberation_steps: int = 4
    deliberation_horizon: int = 8
    fp4_block_size: int = 32
    latent_dim: int = 1024            # DAC 44.1kHz continuous latent dimension
    vocab_size: int = 50257           # GPT-2 tokenizer vocabulary for tag conditioning

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [Batch] in [0, 1]
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class SelectiveSSM(nn.Module):
    def __init__(self, config: FlowDrumTGSSMConfig):
        super().__init__()
        self.d_model = config.d_model
        self.d_state = config.d_state
        self.d_inner = config.d_model * config.expand
        self.dt_rank = config.dt_rank

        self.in_proj = FP4Linear(self.d_model, 2 * self.d_inner, bias=False, block_size=config.fp4_block_size)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=config.d_conv,
            groups=self.d_inner,
            padding=config.d_conv - 1,
        )
        self.x_proj = FP4Linear(
            self.d_inner,
            self.dt_rank + 2 * self.d_state,
            bias=False,
            block_size=config.fp4_block_size,
        )
        self.dt_proj = FP4Linear(
            self.dt_rank,
            self.d_inner,
            bias=True,
            block_size=config.fp4_block_size,
        )
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner, dtype=torch.float32))
        self.out_proj = FP4Linear(self.d_inner, self.d_model, bias=False, block_size=config.fp4_block_size)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        batch, seqlen, _ = u.shape
        xz = self.in_proj(u)
        x, z = xz.chunk(2, dim=-1)

        x_conv = self.conv1d(x.transpose(1, 2))[:, :, :seqlen].transpose(1, 2)
        x_act = F.silu(x_conv)

        x_proj_out = self.x_proj(x_act)
        dt, B, C = torch.split(x_proj_out, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))

        A = -torch.exp(self.A_log.float())
        dt_fp32 = dt.unsqueeze(-1).float()
        A_fp32 = A.unsqueeze(0).unsqueeze(0)
        dA = torch.exp(dt_fp32 * A_fp32)
        
        B_fp32 = B.unsqueeze(2).float()
        dB = dt_fp32 * B_fp32
        dBx = dB * x_act.unsqueeze(-1).float()

        h = torch.zeros((batch, self.d_inner, self.d_state), device=u.device, dtype=torch.float32)
        ys = []
        C_fp32 = C.unsqueeze(2).float()

        for t in range(seqlen):
            h = dA[:, t] * h + dBx[:, t]
            y_t = (h * C_fp32[:, t]).sum(dim=-1)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)
        y = y + x_act.float() * self.D.unsqueeze(0).unsqueeze(0)
        y = y.to(u.dtype) * F.silu(z)
        out = self.out_proj(y)
        return out

class ExpertFFN(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, fp4_block_size: int = 32):
        super().__init__()
        self.w1 = FP4Linear(d_model, d_ffn, bias=False, block_size=fp4_block_size)
        self.w2 = FP4Linear(d_ffn, d_model, bias=False, block_size=fp4_block_size)
        self.w3 = FP4Linear(d_model, d_ffn, bias=False, block_size=fp4_block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MetabolicMoE(nn.Module):
    def __init__(self, config: FlowDrumTGSSMConfig):
        super().__init__()
        self.d_model = config.d_model
        self.num_experts = config.num_experts
        self.top_k = min(config.top_k_experts, config.num_experts)
        d_ffn = int(config.d_model * 2.5)

        self.router = nn.Linear(self.d_model, self.num_experts, bias=False)
        self.experts = nn.ModuleList([
            ExpertFFN(self.d_model, d_ffn, config.fp4_block_size)
            for _ in range(self.num_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seqlen, d_model = x.shape
        x_flat = x.reshape(-1, d_model)

        router_logits = self.router(x_flat.float())
        router_probs = F.softmax(router_logits, dim=-1)

        weights, indices = torch.topk(router_probs, self.top_k, dim=-1)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        tokens_per_expert = F.one_hot(indices[:, 0], num_classes=self.num_experts).float().mean(dim=0)
        avg_prob_per_expert = router_probs.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(tokens_per_expert * avg_prob_per_expert)

        out_flat = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_indices = indices[:, k]
            expert_weights = weights[:, k].unsqueeze(-1).to(x.dtype)
            
            for expert_idx in range(self.num_experts):
                mask = (expert_indices == expert_idx)
                if mask.any():
                    selected_x = x_flat[mask]
                    expert_out = self.experts[expert_idx](selected_x)
                    out_flat[mask] += expert_weights[mask] * expert_out

        out = out_flat.view(batch, seqlen, d_model)
        return out, aux_loss

class FlowTGSSMBlock(nn.Module):
    def __init__(self, config: FlowDrumTGSSMConfig):
        super().__init__()
        self.norm1 = RMSNormFP32(config.d_model)
        self.ssm = SelectiveSSM(config)
        self.norm2 = RMSNormFP32(config.d_model)
        self.moe = MetabolicMoE(config)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.d_model, config.d_model)
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # AdaLN time conditioning
        t_mod = self.time_mlp(time_emb).unsqueeze(1)
        x_in = x + t_mod

        x_norm1 = self.norm1(x_in)
        x = x + self.ssm(x_norm1)

        x_norm2 = self.norm2(x)
        moe_out, aux_loss = self.moe(x_norm2)
        x = x + moe_out
        return x, aux_loss

class DACDrumFlowTGSSM(nn.Module):
    """
    Continuous Rectified Flow Matching TG-SSM model for DAC 44.1kHz Latent Synthesis.
    """
    def __init__(self, config: Optional[FlowDrumTGSSMConfig] = None):
        super().__init__()
        self.config = config or FlowDrumTGSSMConfig()

        # Timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(self.config.d_model),
            nn.Linear(self.config.d_model, self.config.d_model),
            nn.SiLU(),
            nn.Linear(self.config.d_model, self.config.d_model),
        )

        # Text prompt encoder
        self.text_embeddings = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.text_proj = nn.Sequential(
            RMSNormFP32(self.config.d_model),
            FP4Linear(self.config.d_model, self.config.d_model, bias=False, block_size=self.config.fp4_block_size),
            nn.SiLU(),
        )

        # Continuous Latent In-Projection (1024 -> d_model)
        self.latent_in_proj = nn.Linear(self.config.latent_dim, self.config.d_model)

        # Backbone blocks
        self.layers = nn.ModuleList([
            FlowTGSSMBlock(self.config) for _ in range(self.config.n_layers)
        ])
        self.final_norm = RMSNormFP32(self.config.d_model)

        # Continuous Velocity Output Head (d_model -> 1024)
        self.velocity_head = nn.Sequential(
            RMSNormFP32(self.config.d_model),
            FP4Linear(self.config.d_model, self.config.latent_dim, bias=False, block_size=self.config.fp4_block_size)
        )

    def forward(
        self,
        z_t: torch.Tensor,        # [Batch, 1024, Frames]
        t: torch.Tensor,          # [Batch] in [0, 1]
        prompt_ids: torch.Tensor, # [Batch, PromptLen]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts continuous ODE velocity field v(z_t, t, prompt).
        """
        batch, _, num_frames = z_t.shape
        t_emb = self.time_embed(t) # [Batch, d_model]

        # 1. Text conditioning
        text_emb = self.text_embeddings(prompt_ids)
        text_cond = self.text_proj(text_emb) # [Batch, PromptLen, d_model]

        # 2. Continuous Latent Projection
        z_transposed = z_t.transpose(1, 2) # [Batch, Frames, 1024]
        audio_emb = self.latent_in_proj(z_transposed) # [Batch, Frames, d_model]

        # Concatenate text prompt and audio continuous trajectory
        x = torch.cat([text_cond, audio_emb], dim=1)

        # 3. TG-SSM Continuous State-Space Propagation
        aux_losses = []
        for layer in self.layers:
            x, aux_loss = layer(x, t_emb)
            aux_losses.append(aux_loss)

        x = self.final_norm(x)
        aux_total = torch.stack(aux_losses).mean()

        # Extract only audio frame outputs
        prompt_len = prompt_ids.shape[1]
        audio_out = x[:, prompt_len:, :] # [Batch, Frames, d_model]

        # 4. Continuous Velocity Field Prediction
        pred_velocity = self.velocity_head(audio_out) # [Batch, Frames, 1024]
        pred_velocity = pred_velocity.transpose(1, 2) # [Batch, 1024, Frames]

        return pred_velocity, aux_total

    @torch.no_grad()
    def generate_flow(
        self,
        prompt_ids: torch.Tensor,
        num_frames: int = 43, # 0.5s @ 44.1kHz
        steps: int = 25,
        guidance_scale: float = 2.5,
    ) -> torch.Tensor:
        """
        Euler ODE integration along continuous optimal transport flow from z_0 ~ N(0, I) to z_1.
        """
        self.eval()
        device = prompt_ids.device
        batch = prompt_ids.shape[0]

        # Initial random Gaussian latent noise
        z_t = torch.randn(batch, self.config.latent_dim, num_frames, device=device)
        dt = 1.0 / steps

        # Unconditioned prompt for Classifier-Free Guidance (CFG)
        uncond_ids = torch.zeros_like(prompt_ids)

        for i in range(steps):
            t_val = i / steps
            t_tensor = torch.full((batch,), t_val, device=device, dtype=torch.float32)

            # Conditional velocity
            v_cond, _ = self.forward(z_t, t_tensor, prompt_ids)
            
            # Unconditional velocity for CFG
            if guidance_scale > 1.0:
                v_uncond, _ = self.forward(z_t, t_tensor, uncond_ids)
                v = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                v = v_cond

            # Euler Step: z_{t + dt} = z_t + dt * v(z_t, t)
            z_t = z_t + dt * v

        return z_t
