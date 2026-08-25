"""
train_dac_drum_flow.py
Continuous Flow-Matching (Rectified Flow) Training Pipeline for Multi-Genre 44.1kHz Drum Synthesis.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import soundfile as sf
import dac

# Add current workspace to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dac_drum_flow_tgssm import DACDrumFlowTGSSM, FlowDrumTGSSMConfig
from multi_genre_drum_dataset import MultiGenreDrumDataset, TARGET_SR

def train_flow_model(
    max_samples: int = 3000,
    epochs: int = 35,
    start_epoch: int = 1,
    batch_size: int = 16,
    grad_accum: int = 2,
    lr: float = 4e-4,
    device: str = "cuda",
    resume_ckpt: str = None,
):
    print("=" * 75)
    print("🌊 TG-SSM Multi-Genre Continuous Flow-Matching (44.1kHz Studio Drums)")
    print("=" * 75)

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    torch.cuda.empty_cache()

    # 1. Multi-Genre Dataset Loader (Hip-Hop, Trap, Acoustic Rock, Jazz, Funk, Synthwave, Latin)
    dataset = MultiGenreDrumDataset(max_samples=max_samples, device=device)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    print(f"Dataset: {len(dataset):,} samples | Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # 2. Model & DAC Vocoder Setup
    config = FlowDrumTGSSMConfig(
        d_model=384,
        n_layers=6,
        d_state=32,
        expand=2,
        num_experts=4,
        top_k_experts=2,
        latent_dim=1024,
    )
    model = DACDrumFlowTGSSM(config).to(device)

    best_val_loss = float("inf")
    if resume_ckpt and os.path.exists(resume_ckpt):
        print(f"Resuming model weights from {resume_ckpt}...")
        ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 10) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        print(f"Resumed from Epoch {start_epoch-1} (Best Val Loss: {best_val_loss:.5f})")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Initialized Flow-Matching DrumTGSSM: {total_params / 1e6:.2f}M Parameters")

    dac_path = dac.utils.download(model_type="44khz")
    codec = dac.DAC.load(dac_path).to(device)
    codec.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    for _ in range(1, start_epoch):
        scheduler.step()

    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("generated_audio", exist_ok=True)

    # Multi-Genre Test Prompts for Milestone Monitoring
    eval_prompts = [
        "trap, 808, sub kick, deep sub, deep 50Hz sub, heavy low end, hard transient click, resonant sub bass",
        "acoustic rock, snare drum, live studio crack, bright top end, crisp sheen, hard transient click",
        "synthwave, 80s retro, gated snare, analog saturation, diffused room reverb, punchy mid",
        "latin percussion, afrobeat, cowbell, woodblock, bright top end, tight gated, metallic",
    ]

    print(f"\n🚀 Starting Continuous Optimal Transport Flow Matching Training...")
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()
        optimizer.zero_grad()

        for batch_idx, (prompt_ids, continuous_z, _) in enumerate(train_loader):
            prompt_ids = prompt_ids.to(device)
            z_1 = continuous_z.to(device) # [B, 1024, Frames] clean target
            batch_sz, _, num_frames = z_1.shape

            # Sample random noise z_0 ~ N(0, I) and timestep t in [0, 1]
            z_0 = torch.randn_like(z_1)
            t = torch.rand(batch_sz, device=device, dtype=torch.float32)

            # Continuous straight flow interpolation: z_t = t * z_1 + (1 - t) * z_0
            t_expand = t.view(batch_sz, 1, 1)
            z_t = t_expand * z_1 + (1.0 - t_expand) * z_0
            target_velocity = z_1 - z_0

            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                pred_velocity, aux_loss = model(z_t, t, prompt_ids)
                flow_loss = F.mse_loss(pred_velocity, target_velocity)
                loss = (flow_loss + 0.05 * aux_loss) / grad_accum

            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += flow_loss.item()

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        epoch_time = time.time() - t0

        # Validation Pass
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for prompt_ids, continuous_z, _ in val_loader:
                prompt_ids = prompt_ids.to(device)
                z_1 = continuous_z.to(device)
                batch_sz = z_1.shape[0]

                z_0 = torch.randn_like(z_1)
                t = torch.rand(batch_sz, device=device, dtype=torch.float32)
                t_expand = t.view(batch_sz, 1, 1)
                z_t = t_expand * z_1 + (1.0 - t_expand) * z_0
                target_velocity = z_1 - z_0

                with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                    pred_velocity, aux_loss = model(z_t, t, prompt_ids)
                    flow_loss = F.mse_loss(pred_velocity, target_velocity)
                    val_loss += flow_loss.item()

        avg_val_loss = val_loss / len(val_loader)
        lr_curr = scheduler.get_last_lr()[0]

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_time:4.1f}s) | "
            f"Train MSE: {avg_train_loss:7.5f} | "
            f"Val MSE: {avg_val_loss:7.5f} | LR: {lr_curr:.2e}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = "checkpoints/dac_drum_flow_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": config,
                "val_loss": avg_val_loss,
            }, ckpt_path)
            print(f"  ⭐️ Saved Best Checkpoint -> {ckpt_path} (Val MSE: {avg_val_loss:7.5f})")

        # Periodically synthesize multi-genre 44.1kHz studio drum waveforms
        if epoch % 5 == 0 or epoch == epochs:
            print(f"\n🎧 [Epoch {epoch}] Multi-Genre ODE Flow Synthesis & 44.1kHz Decoding...")
            with torch.no_grad():
                for idx, p_text in enumerate(eval_prompts):
                    p_tokens = dataset.tokenizer.encode(
                        p_text,
                        max_length=28,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    ).to(device)

                    gen_z = model.generate_flow(p_tokens, num_frames=43, steps=25, guidance_scale=2.5)
                    audio_44k = codec.decode(gen_z).squeeze().cpu().numpy()
                    audio_44k = (audio_44k / (np.max(np.abs(audio_44k)) + 1e-6)) * 0.95
                    sample_fn = f"generated_audio/multi_genre_epoch_{epoch:02d}_sample_{idx+1}.wav"
                    sf.write(sample_fn, audio_44k, TARGET_SR)
                    print(f"   • Genre Tag Prompt: \"{p_text[:45]}...\" -> {sample_fn}")
            print()

    print("🎉 Multi-Genre Continuous Flow-Matching Training Complete!")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    train_flow_model(
        max_samples=3000,
        epochs=35,
        batch_size=16,
        grad_accum=2,
        lr=4e-4,
        resume_ckpt=args.resume if args.resume and os.path.exists(args.resume) else None,
    )
