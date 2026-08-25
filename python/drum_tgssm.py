"""
drum_tgssm.py
Continuous Selective State-Space Model (TG-SSM) adapted for Text-to-Drum Waveform Synthesis.

Pipeline:
1. Text / Tag conditioning: BPE / Tag Embedding Encoder.
2. Continuous Temporal Backbone: 8-Layer Selective SSM + Metabolic MoE.
3. System 2 Hamiltonian Deliberation Core: Refines acoustic transient trajectories.
4. Multi-Codebook Prediction Head: Projects latent states to 8 EnCodec RVQ codebook logits (1024 classes each).
5. Neural Decoder (EnCodec 24kHz): Inverts RVQ latents into high-fidelity 24kHz studio audio waveforms.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from engine.mixed_precision import FP4Linear, RMSNormFP32

@dataclass
class DrumTGSSMConfig:
    d_model: int = 512                # Latent hidden dimension
    n_layers: int = 8                 # Number of TG-SSM blocks
    d_state: int = 32                 # SSM state expansion
    d_conv: int = 4                   # 1D causal conv kernel
    expand: int = 2                   # SSM expansion factor (d_inner = 1024)
    dt_rank: int = 32                 # Delta rank
    num_experts: int = 4              # MoE experts
    top_k_experts: int = 2            # Top-k active experts
    deliberation_steps: int = 4       # System 2 ODE steps
    deliberation_horizon: int = 8     # Future trajectory rollout
    fp4_block_size: int = 32          # Micro-scaling
    num_codebooks: int = 8            # EnCodec 6.0 kbps RVQ codebooks
    codebook_size: int = 1024         # Classes per codebook
    max_audio_frames: int = 64        # ~0.85 sec at 75 Hz frame rate
    vocab_size: int = 50257           # GPT-2 tokenizer vocabulary for tag conditioning

class SelectiveSSM(nn.Module):
    def __init__(self, config: DrumTGSSMConfig):
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
    def __init__(self, config: DrumTGSSMConfig):
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

class DrumTGSSMBlock(nn.Module):
    def __init__(self, config: DrumTGSSMConfig):
        super().__init__()
        self.norm1 = RMSNormFP32(config.d_model)
        self.ssm = SelectiveSSM(config)
        self.norm2 = RMSNormFP32(config.d_model)
        self.moe = MetabolicMoE(config)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_norm1 = self.norm1(x)
        x = x + self.ssm(x_norm1)

        x_norm2 = self.norm2(x)
        moe_out, aux_loss = self.moe(x_norm2)
        x = x + moe_out
        return x, aux_loss

class LatentDeliberationCore(nn.Module):
    """System 2 Deliberation Core refining transient dynamics on audio manifold."""
    def __init__(self, config: DrumTGSSMConfig):
        super().__init__()
        self.d_model = config.d_model
        self.deliberation_steps = config.deliberation_steps
        self.horizon = config.deliberation_horizon

        self.metric_field = nn.Sequential(
            RMSNormFP32(self.d_model),
            FP4Linear(self.d_model, self.d_model, bias=True, block_size=config.fp4_block_size),
            nn.SiLU(),
            FP4Linear(self.d_model, self.d_model, bias=False, block_size=config.fp4_block_size),
        )
        self.propagator = nn.Sequential(
            RMSNormFP32(self.d_model * 2),
            FP4Linear(self.d_model * 2, self.d_model * 2, bias=True, block_size=config.fp4_block_size),
            nn.SiLU(),
            FP4Linear(self.d_model * 2, self.d_model, bias=False, block_size=config.fp4_block_size),
        )
        self.dt = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, current_z: torch.Tensor, horizon: Optional[int] = None) -> torch.Tensor:
        H = horizon or self.horizon
        v = self.metric_field(current_z)
        z = current_z
        rollout = []
        dt = torch.clamp(F.softplus(self.dt), min=0.01, max=0.5)

        for _ in range(H):
            for _ in range(self.deliberation_steps):
                zv = torch.cat([z, v], dim=-1)
                force = self.propagator(zv)
                v = v + dt * force
            z = z + dt * v
            rollout.append(z)

        return torch.stack(rollout, dim=1)

class DrumTGSSM(nn.Module):
    """Complete Text-to-Drum Waveform Synthesis Model with TG-SSM Core."""
    def __init__(self, config: Optional[DrumTGSSMConfig] = None):
        super().__init__()
        self.config = config or DrumTGSSMConfig()

        # Text prompt encoder
        self.text_embeddings = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.text_proj = nn.Sequential(
            RMSNormFP32(self.config.d_model),
            FP4Linear(self.config.d_model, self.config.d_model, bias=False, block_size=self.config.fp4_block_size),
            nn.SiLU(),
        )

        # RVQ Multi-Codebook Input Embedding: 8 codebooks embedded and summed
        self.rvq_embeddings = nn.ModuleList([
            nn.Embedding(self.config.codebook_size, self.config.d_model)
            for _ in range(self.config.num_codebooks)
        ])
        
        # Audio Frame Positional Embedding
        self.frame_pos_emb = nn.Embedding(self.config.max_audio_frames, self.config.d_model)

        # TGSSM Backbone Blocks
        self.layers = nn.ModuleList([
            DrumTGSSMBlock(self.config) for _ in range(self.config.n_layers)
        ])
        self.final_norm = RMSNormFP32(self.config.d_model)

        # System 2 Deliberator
        self.deliberation_core = LatentDeliberationCore(self.config)

        # Multi-Codebook RVQ Prediction Heads (8 heads predicting 1024 classes each)
        self.rvq_heads = nn.ModuleList([
            FP4Linear(self.config.d_model, self.config.codebook_size, bias=False, block_size=self.config.fp4_block_size)
            for _ in range(self.config.num_codebooks)
        ])

    def forward(
        self,
        prompt_ids: torch.Tensor,              # [Batch, PromptLen]
        audio_codes: Optional[torch.Tensor] = None, # [Batch, NumCodebooks, AudioFrames]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            prompt_ids: [Batch, PromptLen] tokenized text tags
            audio_codes: [Batch, 8, AudioFrames] target/input RVQ audio tokens
        Returns:
            logits: [Batch, 8, AudioFrames, 1024]
            aux_loss: MoE load balancing auxiliary loss
        """
        batch = prompt_ids.shape[0]

        # 1. Text Prompt Encoding
        text_emb = self.text_embeddings(prompt_ids)  # [B, PromptLen, d_model]
        text_cond = self.text_proj(text_emb)

        if audio_codes is not None:
            num_codebooks, num_frames = audio_codes.shape[1], audio_codes.shape[2]
            
            # Embed each of the 8 codebooks and sum
            audio_emb = torch.zeros((batch, num_frames, self.config.d_model), device=prompt_ids.device)
            for q in range(self.config.num_codebooks):
                audio_emb += self.rvq_embeddings[q](audio_codes[:, q, :])
                
            # Add positional embeddings
            pos_indices = torch.arange(num_frames, device=prompt_ids.device).unsqueeze(0).repeat(batch, 1)
            audio_emb += self.frame_pos_emb(pos_indices)

            # Concatenate [Text Conditioning, Audio Sequence]
            full_seq = torch.cat([text_cond, audio_emb], dim=1)
        else:
            full_seq = text_cond

        # 2. Forward through TGSSM Blocks
        aux_losses = []
        x = full_seq
        for layer in self.layers:
            x, aux_loss = layer(x)
            aux_losses.append(aux_loss)

        x = self.final_norm(x)
        aux_loss_total = torch.stack(aux_losses).mean()

        # Extract only the audio frame latent representations (skip prompt tokens)
        prompt_len = prompt_ids.shape[1]
        audio_latents = x[:, prompt_len:, :]  # [Batch, AudioFrames, d_model]

        # 3. Project to 8 RVQ Codebook Logits
        head_logits = []
        for q in range(self.config.num_codebooks):
            logits_q = self.rvq_heads[q](audio_latents)  # [Batch, AudioFrames, 1024]
            head_logits.append(logits_q)

        # [Batch, 8, AudioFrames, 1024]
        rvq_logits = torch.stack(head_logits, dim=1)
        return rvq_logits, aux_loss_total

    @torch.no_grad()
    def generate_codes(
        self,
        prompt_ids: torch.Tensor,
        num_frames: int = 48,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        Autoregressive / Parallel continuous rollout to generate [1, 8, num_frames] RVQ codes.
        """
        self.eval()
        device = prompt_ids.device
        batch = prompt_ids.shape[0]

        curr_codes = torch.zeros((batch, self.config.num_codebooks, 0), dtype=torch.long, device=device)

        for step in range(num_frames):
            if curr_codes.shape[-1] == 0:
                # Seed with dummy zero frame
                dummy = torch.zeros((batch, self.config.num_codebooks, 1), dtype=torch.long, device=device)
                logits, _ = self.forward(prompt_ids=prompt_ids, audio_codes=dummy)
            else:
                logits, _ = self.forward(prompt_ids=prompt_ids, audio_codes=curr_codes)

            # Sample next frame for all 8 codebooks
            next_frame_codes = []
            for q in range(self.config.num_codebooks):
                head_l = logits[:, q, -1, :] / max(1e-4, temperature)
                if top_k > 0:
                    v, _ = torch.topk(head_l, min(top_k, head_l.size(-1)))
                    head_l[head_l < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(head_l, dim=-1)
                tok = torch.multinomial(probs, num_samples=1)  # [B, 1]
                next_frame_codes.append(tok)

            next_frame = torch.stack(next_frame_codes, dim=1)  # [B, 8, 1]
            curr_codes = torch.cat([curr_codes, next_frame], dim=-1)

        return curr_codes
