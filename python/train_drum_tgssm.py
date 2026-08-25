"""
train_drum_tgssm.py
Train DrumTGSSM on Real Hugging Face Studio Drum Dataset (yojul/one-shot-hip-hop-drums) on RTX 3060.
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import soundfile as sf
import encodec

# Add current workspace to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drum_tgssm import DrumTGSSM, DrumTGSSMConfig
from real_drum_dataset import RealDrumDataset, TARGET_SR

def train_real_drum_model(
    max_samples: int = 3000,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 4e-4,
    device: str = "cuda",
):
    print("=" * 75)
    print("🥁 TG-SSM Real Drum Sample Waveform Generator — RTX 3060 Training")
    print("=" * 75)

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    # 1. Dataset from Hugging Face
    dataset = RealDrumDataset(max_samples=max_samples, device=device)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    print(f"Dataset: {len(dataset):,} real drum samples | Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # 2. Model & Codec Setup
    config = DrumTGSSMConfig(
        d_model=384,
        n_layers=6,
        d_state=32,
        expand=2,
        num_experts=4,
        top_k_experts=2,
        num_codebooks=8,
        codebook_size=1024,
        max_audio_frames=64,
    )
    model = DrumTGSSM(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Initialized DrumTGSSM: {total_params / 1e6:.2f}M Parameters (FP4 & FP32 Pinned Layers)")

    codec = encodec.EncodecModel.encodec_model_24khz().to(device)
    codec.set_target_bandwidth(6.0)
    codec.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("generated_drums", exist_ok=True)
    best_val_loss = float("inf")

    eval_prompts = [
        "808, sub kick, sub bass, deep, deep low end, heavy sub, hard transient, 50Hz sub punch",
        "snare, drum, acoustic snare, bright, crisp, hard transient, snappy wire rattle",
        "hihat, closed hat, metallic, crisp, fast decay, short tail, trap sizzle",
        "clap, handclap, percussion, warm, punchy mid, layered, room resonance",
    ]

    print("\n🚀 Starting High-Throughput Training Loop on RTX 3060...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_ce = 0.0
        train_aux = 0.0
        t0 = time.time()

        for batch_idx, (prompt_ids, audio_codes, _) in enumerate(train_loader):
            prompt_ids = prompt_ids.to(device)
            audio_codes = audio_codes.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                logits, aux_loss = model(prompt_ids, audio_codes)
                B, Q, T, C = logits.shape
                logits_flat = logits.view(-1, C)
                targets_flat = audio_codes.view(-1)
                ce_loss = F.cross_entropy(logits_flat, targets_flat)
                total_loss = ce_loss + 0.05 * aux_loss

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += total_loss.item()
            train_ce += ce_loss.item()
            train_aux += aux_loss.item()

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        avg_train_ce = train_ce / len(train_loader)
        epoch_time = time.time() - t0

        # Validation Pass
        model.eval()
        val_loss = 0.0
        val_ce = 0.0
        with torch.no_grad():
            for prompt_ids, audio_codes, _ in val_loader:
                prompt_ids = prompt_ids.to(device)
                audio_codes = audio_codes.to(device)
                with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                    logits, aux_loss = model(prompt_ids, audio_codes)
                    B, Q, T, C = logits.shape
                    ce_loss = F.cross_entropy(logits.view(-1, C), audio_codes.view(-1))
                    val_loss += (ce_loss + 0.05 * aux_loss).item()
                    val_ce += ce_loss.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_ce = val_ce / len(val_loader)

        lr_curr = scheduler.get_last_lr()[0]
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_time:4.1f}s) | "
            f"Train Loss: {avg_train_loss:6.4f} (CE: {avg_train_ce:6.4f}) | "
            f"Val Loss: {avg_val_loss:6.4f} (CE: {avg_val_ce:6.4f}) | LR: {lr_curr:.2e}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = "checkpoints/drum_tgssm_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": config,
                "val_loss": avg_val_loss,
            }, ckpt_path)
            print(f"  ⭐️ Saved Best Checkpoint -> {ckpt_path} (Val Loss: {avg_val_loss:.4f})")

        # Periodically synthesize test drum samples from text tags
        if epoch % 10 == 0 or epoch == epochs:
            print(f"\n🎧 [Epoch {epoch}] Synthesizing Real Drum Waveforms from Text Tags...")
            with torch.no_grad():
                for idx, p_text in enumerate(eval_prompts):
                    p_tokens = dataset.tokenizer.encode(
                        p_text,
                        max_length=24,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    ).to(device)

                    gen_codes = model.generate_codes(p_tokens, num_frames=48, temperature=0.75)
                    frame_tuples = [(gen_codes, None)]
                    wav_decoded = codec.decode(frame_tuples).squeeze(0).squeeze(0).cpu().numpy()
                    
                    sample_fn = f"generated_drums/real_epoch_{epoch:02d}_sample_{idx+1}.wav"
                    sf.write(sample_fn, wav_decoded, TARGET_SR)
                    print(f"   • Tag Prompt: \"{p_text[:45]}...\" -> {sample_fn}")
            print()

    print("🎉 Real Drum TG-SSM Training Complete!")
    return model

if __name__ == "__main__":
    train_real_drum_model(max_samples=3000, epochs=30, batch_size=32, lr=4e-4)
