"""
generate_dac_drum.py
High-Fidelity 44.1kHz Continuous Flow-Matching Studio Drum Synthesizer.
Supports 13 Production ODE/SDE Solvers Borrowed from Stable Audio & Flux.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import soundfile as sf
from transformers import AutoTokenizer
import dac

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dac_drum_flow_tgssm import DACDrumFlowTGSSM, FlowDrumTGSSMConfig

SAMPLER_CHOICES = [
    "heun",
    "euler",
    "dpmpp_2m",
    "dpmpp_2s",
    "euler_ancestral",
    "heun_ancestral",
    "rk4",
    "midpoint",
    "bogacki_shampine",
    "dopri5",
    "flux_rf",
    "stable_audio_euler",
    "lms"
]

def generate_studio_drum(
    prompt: str,
    output_path: str = "generated_audio/heun/drum_sample.wav",
    checkpoint_path: str = "checkpoints/dac_drum_flow_best.pt",
    steps: int = 30,
    guidance_scale: float = 3.0,
    sampler: str = "heun",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    eta: float = 0.2,
):
    print("=" * 70)
    print("🌊 TG-SSM 44.1kHz Multi-Sampler Studio Drum Synthesis")
    print("=" * 70)
    print(f"📥 Input Tag Prompt: \"{prompt}\"")
    print(f"🔬 ODE/SDE Sampler:  {sampler.upper()} ({sampler})")
    print(f"⚙️  Compute Device:   {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    prompt_ids = tokenizer.encode(
        prompt,
        max_length=28,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)

    # 2. Load Model Checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", FlowDrumTGSSMConfig())
    model = DACDrumFlowTGSSM(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 3. Load DAC Vocoder
    dac_path = dac.utils.download(model_type="44khz")
    codec = dac.DAC.load(dac_path).to(device)
    codec.eval()

    # 4. Continuous Optimal Transport Flow Generation
    t0 = time.time()
    print(f"\n⚡ Solving Continuous Optimal Transport Flow ({steps} steps, CFG={guidance_scale}, Sampler={sampler.upper()})...")
    with torch.no_grad():
        gen_z = model.generate_flow(
            prompt_ids,
            num_frames=43, # 43 frames @ 44.1kHz = 22,016 audio samples (0.50 seconds)
            steps=steps,
            guidance_scale=guidance_scale,
            sampler=sampler,
            eta=eta,
        )
    flow_time = (time.time() - t0) * 1000
    print(f"✅ Generated Continuous Manifold Trajectory: {list(gen_z.shape)} in {flow_time:.1f} ms")

    # 5. Decode with DAC Neural Vocoder into 44.1kHz Studio Waveform
    t1 = time.time()
    print(f"🔊 Decoding 1024-dim Continuous Latent into 44.1kHz Studio Waveform...")
    with torch.no_grad():
        audio = codec.decode(gen_z).squeeze().cpu().numpy()

    # Peak normalization to -0.5 dBFS
    peak = np.max(np.abs(audio)) + 1e-6
    audio = (audio / peak) * 0.95
    decode_time = (time.time() - t1) * 1000

    # 6. Save Waveform
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, audio, 44100)

    total_time = flow_time + decode_time
    print(f"🎉 Pristine Studio Drum Waveform Saved: {output_path}")
    print(f"📊 Audio Quality:   44100Hz Studio Master ({len(audio):,} samples, {len(audio)/44100:.2f}s)")
    print(f"⏱️  Inference Time:  {total_time:.1f} ms (Flow: {flow_time:.1f}ms | Neural Dec: {decode_time:.1f}ms)")
    print("=" * 70)
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TG-SSM 44.1kHz Continuous Flow-Matching Multi-Sampler Drum Synthesizer")
    parser.add_argument("--prompt", type=str, default="trap, 808, sub kick, deep sub, deep 50Hz sub, heavy low end, hard transient click")
    parser.add_argument("--output", type=str, default="generated_audio/heun/test_sample.wav")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/dac_drum_flow_best.pt")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--sampler", type=str, default="heun", choices=SAMPLER_CHOICES, help="13 ODE/SDE flow solvers (Stable Audio & Flux)")
    parser.add_argument("--eta", type=float, default=0.2, help="Langevin noise churn for stochastic SDE samplers")
    args = parser.parse_args()

    generate_studio_drum(
        prompt=args.prompt,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        steps=args.steps,
        guidance_scale=args.cfg,
        sampler=args.sampler,
        eta=args.eta,
    )
