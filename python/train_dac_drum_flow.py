"""
train_dac_drum_flow.py
Continuous Flow-Matching Training Pipeline for Multi-Genre Drum Synthesis.
Supports resuming from Epoch 60 and training until convergence.
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

def train_until_convergence(
    max_samples: int = 5000,
    add_epochs: int = 50,
    batch_size: int = 16,
    grad_accum: int = 2,
    lr: float = 1.5e-4,
    device: str = "cuda",
    resume_ckpt: str = "checkpoints/dac_drum_flow_best.pt",
    patience: int = 15,
):
    print("=" * 75)
    print("🌊 TG-SSM Convergence Training (Epoch 61+ Multi-Genre Studio Drums)")
    print("=" * 75)

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    torch.cuda.empty_cache()

    # 1. Load Broadened 5,000-Sample Dataset
    dataset = MultiGenreDrumDataset(max_samples=max_samples, device=device)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    print(f"Dataset: {len(dataset):,} samples | Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # 2. Model Setup
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

    start_epoch = 1
    best_val_loss = float("inf")
    
    if resume_ckpt and os.path.exists(resume_ckpt):
        print(f"Resuming model weights from {resume_ckpt}...")
        ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        past_epoch = ckpt.get("epoch", 60)
        start_epoch = past_epoch + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        print(f"Resumed from Epoch {past_epoch} (Best Val Loss: {best_val_loss:.5f})")

    total_epochs = start_epoch + add_epochs - 1
    print(f"Training Schedule: Epoch [{start_epoch} -> {total_epochs}] (+{add_epochs} epochs | Patience: {patience})")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Initialized Flow-Matching DrumTGSSM: {total_params / 1e6:.2f}M Parameters")

    dac_path = dac.utils.download(model_type="44khz")
    codec = dac.DAC.load(dac_path).to(device)
    codec.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=add_epochs, eta_min=5e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("generated_audio/euler", exist_ok=True)
    os.makedirs("generated_audio/heun", exist_ok=True)

    eval_prompts = [
        ("808", "trap, 808, sub kick, deep sub, deep 50Hz sub, heavy low end, hard transient click, resonant sub bass"),
        ("snare", "acoustic rock, snare drum, live studio crack, bright top end, crisp sheen, hard transient click"),
        ("synthwave", "synthwave, 80s retro, gated snare, analog saturation, diffused room reverb, punchy mid"),
        ("cowbell", "latin percussion, 808 cowbell, metallic cowbell, dual square wave, sharp transient click, bright top end"),
    ]

    patience_counter = 0

    print(f"\n🚀 Starting Continuous Flow-Matching Training until Convergence...")
    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()
        optimizer.zero_grad()

        for batch_idx, (prompt_ids, continuous_z, _) in enumerate(train_loader):
            prompt_ids = prompt_ids.to(device)
            z_1 = continuous_z.to(device)
            batch_sz, _, num_frames = z_1.shape

            z_0 = torch.randn_like(z_1)
            t = torch.rand(batch_sz, device=device, dtype=torch.float32)

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
            f"Epoch [{epoch:03d}/{total_epochs:03d}] ({epoch_time:4.1f}s) | "
            f"Train MSE: {avg_train_loss:7.5f} | "
            f"Val MSE: {avg_val_loss:7.5f} | LR: {lr_curr:.2e}"
        )

        if avg_val_loss < best_val_loss:
            improvement = best_val_loss - avg_val_loss
            best_val_loss = avg_val_loss
            patience_counter = 0
            ckpt_path = "checkpoints/dac_drum_flow_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": config,
                "val_loss": avg_val_loss,
            }, ckpt_path)
            print(f"  ⭐️ Saved Best Checkpoint -> {ckpt_path} (Val MSE: {avg_val_loss:7.5f} | Improvement: -{improvement:.5f})")
        else:
            patience_counter += 1

        # Periodically synthesize 44.1kHz studio drum waveforms with Heun & Euler
        if epoch % 5 == 0 or epoch == total_epochs or patience_counter >= patience:
            print(f"\n🎧 [Epoch {epoch}] Multi-Genre ODE Flow Synthesis & 44.1kHz Decoding (Heun & Euler)...")
            with torch.no_grad():
                for name, p_text in eval_prompts:
                    p_tokens = dataset.tokenizer.encode(
                        p_text,
                        max_length=28,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    ).to(device)

                    # 1. Euler sample
                    gen_z_euler = model.generate_flow(p_tokens, num_frames=43, steps=25, guidance_scale=3.0, sampler="euler")
                    audio_euler = codec.decode(gen_z_euler).squeeze().cpu().numpy()
                    audio_euler = (audio_euler / (np.max(np.abs(audio_euler)) + 1e-6)) * 0.95
                    euler_fn = f"generated_audio/euler/epoch_{epoch:03d}_{name}.wav"
                    sf.write(euler_fn, audio_euler, TARGET_SR)

                    # 2. Heun sample
                    gen_z_heun = model.generate_flow(p_tokens, num_frames=43, steps=25, guidance_scale=3.0, sampler="heun")
                    audio_heun = codec.decode(gen_z_heun).squeeze().cpu().numpy()
                    audio_heun = (audio_heun / (np.max(np.abs(audio_heun)) + 1e-6)) * 0.95
                    heun_fn = f"generated_audio/heun/epoch_{epoch:03d}_{name}.wav"
                    sf.write(heun_fn, audio_heun, TARGET_SR)

                    print(f"   • [{name.upper()}] Euler -> {euler_fn} | Heun -> {heun_fn}")
            print()

        if patience_counter >= patience:
            print(f"🛑 Convergence Plateau Detected (No Val MSE improvement for {patience} consecutive epochs). Training Complete!")
            break

    print("🎉 Continuous Flow-Matching Convergence Training Complete!")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default="checkpoints/dac_drum_flow_best.pt")
    parser.add_argument("--add_epochs", type=int, default=50)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    args = parser.parse_args()

    train_until_convergence(
        max_samples=args.samples,
        add_epochs=args.add_epochs,
        batch_size=16,
        grad_accum=2,
        lr=args.lr,
        resume_ckpt=args.resume if args.resume and os.path.exists(args.resume) else None,
        patience=args.patience,
    )
