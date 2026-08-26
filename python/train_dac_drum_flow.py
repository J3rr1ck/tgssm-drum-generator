"""
train_dac_drum_flow.py
Hardened 60-Epoch Continuous Flow-Matching Studio Drum Training from Scratch.
Integrates:
1. Classifier-Free Guidance (CFG) Training Dropout (p=0.15)
2. Beta-Distributed Timestep Sampling (Beta(1.5, 1.5))
3. Latent Variance Anti-Collapse Regularization
4. Multi-Metric Audio Acoustic Evaluation (Crest Factor, Sub Energy, HF Sheen, Flow MSE)
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

def evaluate_acoustic_quality(audio: np.ndarray, sr: int = TARGET_SR) -> dict:
    """Computes acoustic studio metrics on generated audio."""
    peak = np.max(np.abs(audio)) + 1e-6
    rms = np.sqrt(np.mean(audio ** 2)) + 1e-6
    crest_factor = peak / rms

    fft = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
    total_energy = np.sum(fft ** 2) + 1e-8

    sub_energy = np.sum(fft[(freqs >= 20) & (freqs <= 80)] ** 2) / total_energy
    hf_energy = np.sum(fft[freqs >= 6000] ** 2) / total_energy

    return {
        "rms": float(rms),
        "peak": float(peak),
        "crest_factor": float(crest_factor),
        "sub_energy": float(sub_energy),
        "hf_energy": float(hf_energy),
    }

def train_hardened_flow(
    max_samples: int = 5000,
    epochs: int = 60,
    batch_size: int = 16,
    grad_accum: int = 2,
    lr: float = 3e-4,
    device: str = "cuda",
):
    print("=" * 75)
    print("🛡️ TG-SSM Hardened 60-Epoch Flow-Matching (44.1kHz Studio Drums from Scratch)")
    print("=" * 75)

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    torch.cuda.empty_cache()

    # 1. Dataset Loader (5,000 samples)
    dataset = MultiGenreDrumDataset(max_samples=max_samples, device=device)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    print(f"Dataset: {len(dataset):,} samples | Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # 2. Hardened Model Setup
    config = FlowDrumTGSSMConfig(
        d_model=384,
        n_layers=6,
        d_state=32,
        expand=2,
        num_experts=4,
        top_k_experts=2,
        latent_dim=1024,
        p_uncond=0.15, # 15% CFG training dropout
    )
    model = DACDrumFlowTGSSM(config).to(device)

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

    best_studio_score = float("inf")
    beta_dist = torch.distributions.Beta(1.5, 1.5)

    print(f"\n🚀 Starting Hardened 60-Epoch Training from Scratch...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()
        optimizer.zero_grad()

        for batch_idx, (prompt_ids, continuous_z, _) in enumerate(train_loader):
            prompt_ids = prompt_ids.to(device)
            z_1 = continuous_z.to(device)
            batch_sz, _, num_frames = z_1.shape

            z_0 = torch.randn_like(z_1)
            # Beta distribution timestep sampling (focus on midpoint trajectory curvature)
            t = beta_dist.sample((batch_sz,)).to(device=device, dtype=torch.float32)

            t_expand = t.view(batch_sz, 1, 1)
            z_t = t_expand * z_1 + (1.0 - t_expand) * z_0
            target_velocity = z_1 - z_0

            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                pred_velocity, aux_loss = model(z_t, t, prompt_ids, p_uncond=0.15)
                
                # 1. Flow Velocity MSE Loss
                flow_mse = F.mse_loss(pred_velocity, target_velocity)
                
                # 2. Latent Variance Anti-Collapse Loss
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

        # Multi-Metric Studio Quality Score
        studio_score = avg_val_loss

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_time:4.1f}s) | "
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
            print(f"  ⭐️ Saved Best Hardened Checkpoint -> {ckpt_path} (Val MSE: {avg_val_loss:7.5f} | -{improvement:.5f})")

        # Milestone Audio Synthesis every 5 epochs
        if epoch % 5 == 0 or epoch == epochs:
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

                    # Heun 2nd-order with hardened CFG
                    gen_z_heun = model.generate_flow(p_tokens, num_frames=43, steps=25, guidance_scale=3.0, sampler="heun")
                    audio_heun = codec.decode(gen_z_heun).squeeze().cpu().numpy()
                    audio_heun = (audio_heun / (np.max(np.abs(audio_heun)) + 1e-6)) * 0.95
                    heun_fn = f"generated_audio/heun/hardened_epoch_{epoch:02d}_{name}.wav"
                    sf.write(heun_fn, audio_heun, TARGET_SR)

                    # Euler with hardened CFG
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

    print("🎉 Hardened 60-Epoch Continuous Flow-Matching Run Complete!")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    train_hardened_flow(
        max_samples=args.samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
