"""
generate_dac_drum.py
Interactive CLI for 44.1kHz Studio-Grade Drum Waveform Synthesis via TG-SSM Flow Matching & DAC.
Supports multiple high-order ODE samplers: Heun (2nd order), Midpoint, RK4, and Euler.
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import soundfile as sf
from transformers import AutoTokenizer
import dac

# Add current workspace to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dac_drum_flow_tgssm import DACDrumFlowTGSSM, FlowDrumTGSSMConfig
from dac_flow_dataset import TARGET_SR

def generate_studio_drum(
    prompt: str,
    output_path: str = "generated_audio/studio_808_kick.wav",
    checkpoint_path: str = "checkpoints/dac_drum_flow_best.pt",
    steps: int = 30,
    guidance_scale: float = 3.0,
    sampler: str = "heun",
    device: str = "cuda",
):
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print("=" * 70)
    print("🌊 TG-SSM 44.1kHz Multi-Sampler Studio Drum Synthesis")
    print("=" * 70)
    print(f"📥 Input Tag Prompt: \"{prompt}\"")
    print(f"🔬 ODE Sampler:      {sampler.upper()} ({'2nd-Order Predictor-Corrector' if sampler == 'heun' else sampler})")
    print(f"⚙️  Compute Device:   {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Train the model first!")

    # 1. Load Flow Matching TG-SSM Model
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", FlowDrumTGSSMConfig())
    model = DACDrumFlowTGSSM(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 2. Tokenize Text Tags
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    p_tokens = tokenizer.encode(
        prompt,
        max_length=28,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)

    # 3. Load DAC 44.1kHz Studio Vocoder
    dac_path = dac.utils.download(model_type="44khz")
    codec = dac.DAC.load(dac_path).to(device)
    codec.eval()

    # 4. Continuous ODE Integration (Flow Matching)
    print(f"\n⚡ Solving Continuous Optimal Transport Flow ({steps} steps, CFG={guidance_scale}, Sampler={sampler.upper()})...")
    t0 = time.time()
    with torch.no_grad():
        gen_z = model.generate_flow(
            p_tokens,
            num_frames=43,
            steps=steps,
            guidance_scale=guidance_scale,
            sampler=sampler,
        )
        t_flow = time.time() - t0
        print(f"✅ Generated Continuous Manifold Trajectory: {list(gen_z.shape)} in {t_flow*1000:.1f} ms")

        print("🔊 Decoding 1024-dim Continuous Latent into 44.1kHz Studio Waveform...")
        t_dec0 = time.time()
        audio = codec.decode(gen_z).squeeze().cpu().numpy()
        t_dec = time.time() - t_dec0

    # Normalize audio
    audio = (audio / (np.max(np.abs(audio)) + 1e-6)) * 0.95
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    sf.write(output_path, audio, TARGET_SR)
    dur = len(audio) / TARGET_SR

    print(f"🎉 Pristine Studio Drum Waveform Saved: {output_path}")
    print(f"📊 Audio Quality:   {TARGET_SR}Hz Studio Master ({len(audio):,} samples, {dur:.2f}s)")
    print(f"⏱️  Inference Time:  {(t_flow + t_dec)*1000:.1f} ms (Flow: {t_flow*1000:.1f}ms | Neural Dec: {t_dec*1000:.1f}ms)")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 44.1kHz drum sample from text tags via Flow Matching.")
    parser.add_argument("--prompt", type=str, default="trap, 808, sub kick, deep sub, deep 50Hz sub, heavy low end, hard transient click")
    parser.add_argument("--output", type=str, default="generated_audio/studio_drum.wav")
    parser.add_argument("--ckpt", type=str, default="checkpoints/dac_drum_flow_best.pt")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--sampler", type=str, default="heun", choices=["heun", "euler", "midpoint", "rk4"], help="ODE Sampler")
    args = parser.parse_args()

    generate_studio_drum(
        prompt=args.prompt,
        output_path=args.output,
        checkpoint_path=args.ckpt,
        steps=args.steps,
        guidance_scale=args.cfg,
        sampler=args.sampler,
    )
