"""
train_dac_drum_flow.py
Hardened Training Pipeline for Continuous Flow-Matching TG-SSM 44.1kHz Studio Drum Synthesis.
Features:
- Continual training resume support from best checkpoint
- 15% CFG Training Dropout (p_uncond=0.15)
- Beta(1.5, 1.5) Timestep Distribution Sampling
- Latent Variance Anti-Collapse Loss
- Multi-Metric Acoustic Quality Evaluation (Transient Crest Factor, High-Freq Sheen, Sub-Bass Concentration)
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

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dac_drum_flow_tgssm import DACDrumFlowTGSSM, FlowDrumTGSSMConfig
from dac_flow_dataset import DACContinuousDrumDataset

TARGET_SR = 44100

def evaluate_acoustic_quality(audio: np.ndarray, sr: int = TARGET_SR) -> dict:
    """Calculates quantitative acoustic metrics on generated drum waveforms."""
    peak = np.max(np.abs(audio)) + 1e-8
    rms = np.sqrt(np.mean(audio ** 2)) + 1e-8
    crest_factor = peak / rms

    # Frequency Domain Analysis via FFT
    fft_vals = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
    total_energy = np.sum(fft_vals ** 2) + 1e-8

    # Sub-bass concentration (20Hz - 80Hz)
    sub_mask = (freqs >= 20) & (freqs <= 80)
    sub_energy = np.sum(fft_vals[sub_mask] ** 2) / total_energy

    # High-frequency sheen (> 6000Hz)
    hf_mask = freqs >= 6000
    hf_energy = np.sum(fft_vals[hf_mask] ** 2) / total_energy

    return {
        "crest_factor": float(crest_factor),
        "sub_energy": float(sub_energy),
        "hf_energy": float(hf_energy),
        "rms": float(rms),
    }

def train_hardened_flow(
    max_samples: int = 5000,
    epochs: int = 60,
    batch_size: int = 16,
    grad_accum: int = 2,
    lr: float = 3e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    resume_checkpoint: str = None,
):
    print("=" * 75)
    print("🥁 Hardened 44.1kHz Continuous Flow-Matching TG-SSM Drum Training")
    print("=" * 75)
    print(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    torch.cuda.empty_cache()

    # 1. Dataset Loader
    dataset = DACContinuousDrumDataset(max_samples=max_samples)
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
        p_uncond=0.15,
    )
    model = DACDrumFlowTGSSM(config).to(device)

    start_epoch = 1
    best_studio_score = float("inf")
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"🔄 Resuming from checkpoint: {resume_checkpoint}")
        ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_studio_score = ckpt.get("val_loss", float("inf"))
        print(f"  ⭐️ Successfully loaded weights (Previous Best Val MSE: {best_studio_score:.5f}) | Starting at Epoch {start_epoch}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Initialized Hardened Flow-Matching TGSSM: {total_params / 1e6:.2f}M Parameters")

    dac_path = dac.utils.download(model_type="44khz")
    codec = dac.DAC.load(dac_path).to(device)
    codec.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
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

    beta_dist = torch.distributions.Beta(1.5, 1.5)

    print(f"\n🚀 Starting Continuous Flow Training (Epochs {start_epoch} to {start_epoch + epochs - 1})...")
    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        train_loss = 0.0
        t0 = time.time()
        optimizer.zero_grad()

        for batch_idx, (prompt_ids, continuous_z, _) in enumerate(train_loader):
            prompt_ids = prompt_ids.to(device)
            z_1 = continuous_z.to(device) # Data target
            batch_sz, dim, frames = z_1.shape

            z_0 = torch.randn_like(z_1)
            t = beta_dist.sample((batch_sz,)).to(device=device, dtype=torch.float32)

            t_expand = t.view(batch_sz, 1, 1)
            z_t = t_expand * z_1 + (1.0 - t_expand) * z_0
            target_velocity = z_1 - z_0

            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                pred_velocity, aux_loss = model(z_t, t, prompt_ids, p_uncond=0.15)
                
                # Flow Velocity MSE Loss
                flow_mse = F.mse_loss(pred_velocity, target_velocity)

                # Latent Variance Anti-Collapse Loss
                pred_std = pred_velocity.std(dim=-1)
                target_std = target_velocity.std(dim=-1)
                var_loss = F.mse_loss(pred_std, target_std)
                
                loss = (flow_mse + 0.15 * var_loss + 0.05 * aux_loss) / grad_accum

            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += flow_mse.item()

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        epoch_time = time.time() - t0

        # Validation Pass
        model.eval()
        val_flow_loss = 0.0
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
                    pred_velocity, aux_loss = model(z_t, t, prompt_ids, p_uncond=0.0)
                    flow_mse = F.mse_loss(pred_velocity, target_velocity)
                    val_flow_loss += flow_mse.item()

        avg_val_loss = val_flow_loss / len(val_loader)
        lr_curr = scheduler.get_last_lr()[0]
        studio_score = avg_val_loss

        print(
            f"Epoch [{epoch:02d}/{start_epoch + epochs - 1:02d}] ({epoch_time:4.1f}s) | "
            f"Train MSE: {avg_train_loss:7.5f} | "
            f"Val MSE: {avg_val_loss:7.5f} | LR: {lr_curr:.2e}"
        )

        if studio_score < best_studio_score:
            improvement = best_studio_score - studio_score
            best_studio_score = studio_score
            ckpt_path = "checkpoints/dac_drum_flow_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": config,
                "val_loss": avg_val_loss,
            }, ckpt_path)
            print(f"  ⭐️ Saved Best Hardened Checkpoint -> {ckpt_path} (Val MSE: {best_studio_score:7.5f} | -{improvement:7.5f})")

        # Acoustic Quality Evaluation Every 5 Epochs
        if epoch % 5 == 0 or epoch == (start_epoch + epochs - 1):
            print(f"\n🎧 [Epoch {epoch:02d}] Hardened Multi-Genre Synthesis & Acoustic Quality Evaluation...")
            with torch.no_grad():
                for name, p_text in eval_prompts:
                    p_tokens = dataset.tokenizer.encode(
                        p_text,
                        max_length=28,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    ).to(device)

                    # 1. Generate with Heun 2nd-order Predictor-Corrector
                    gen_z_heun = model.generate_flow(p_tokens, num_frames=43, steps=25, guidance_scale=3.0, sampler="heun")
                    audio_heun = codec.decode(gen_z_heun).squeeze().cpu().numpy()
                    audio_heun = (audio_heun / (np.max(np.abs(audio_heun)) + 1e-6)) * 0.95
                    heun_fn = f"generated_audio/heun/hardened_epoch_{epoch:02d}_{name}.wav"
                    sf.write(heun_fn, audio_heun, TARGET_SR)

                    # 2. Generate with Euler 1st-order Flow
                    gen_z_euler = model.generate_flow(p_tokens, num_frames=43, steps=25, guidance_scale=3.0, sampler="euler")
                    audio_euler = codec.decode(gen_z_euler).squeeze().cpu().numpy()
                    audio_euler = (audio_euler / (np.max(np.abs(audio_euler)) + 1e-6)) * 0.95
                    euler_fn = f"generated_audio/euler/hardened_epoch_{epoch:02d}_{name}.wav"
                    sf.write(euler_fn, audio_euler, TARGET_SR)

                    metrics = evaluate_acoustic_quality(audio_heun, TARGET_SR)
                    print(
                        f"   • [{name.upper():<9}] Heun: {heun_fn} | "
                        f"RMS: {metrics['rms']:.3f} | Crest: {metrics['crest_factor']:4.1f} | "
                        f"Sub: {metrics['sub_energy']*100:4.1f}% | HF: {metrics['hf_energy']*100:4.1f}%"
                    )
            print()

    print("🎉 Convergence Training Run Complete!")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    train_hardened_flow(
        max_samples=args.samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resume_checkpoint=args.resume,
    )
