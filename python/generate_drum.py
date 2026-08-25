"""
generate_drum.py
Interactive CLI for Text-Conditioned Drum Sample Waveform Synthesis via TG-SSM & EnCodec.
"""

import os
import sys
import argparse
import time
import torch
import soundfile as sf
from transformers import AutoTokenizer
import encodec

# Add current workspace to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drum_tgssm import DrumTGSSM, DrumTGSSMConfig
from drum_synth_engine import SAMPLE_RATE

def generate_sample_from_prompt(
    prompt: str,
    output_path: str = "output_drum.wav",
    checkpoint_path: str = "checkpoints/drum_tgssm_best.pt",
    duration_frames: int = 48, # 0.64s @ 75Hz
    temperature: float = 0.75,
    top_k: int = 50,
    device: str = "cuda",
):
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print("=" * 65)
    print("🥁 TG-SSM Text-to-Drum Waveform Synthesis")
    print("=" * 65)
    print(f"📥 Input Tag Prompt: \"{prompt}\"")
    print(f"⚙️  Compute Device:   {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    # 1. Load Checkpoint & Config
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Train the model first!")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", DrumTGSSMConfig())
    model = DrumTGSSM(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 2. Tokenize Prompt
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    p_tokens = tokenizer.encode(
        prompt,
        max_length=24,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)

    # 3. Load EnCodec 24kHz Neural Decoder
    codec = encodec.EncodecModel.encodec_model_24khz().to(device)
    codec.set_target_bandwidth(6.0)
    codec.eval()

    # 4. Latent Rollout & Waveform Generation
    print("\n⚡ Running TG-SSM Continuous Latent State-Space Rollout...")
    t0 = time.time()
    with torch.no_grad():
        gen_codes = model.generate_codes(
            p_tokens,
            num_frames=duration_frames,
            temperature=temperature,
            top_k=top_k
        )
        t_gen = time.time() - t0

        print(f"✅ Generated RVQ Codebook Grid: {list(gen_codes.shape)} in {t_gen*1000:.2f} ms")
        print("🔊 Inverting Latent Tokens via EnCodec 24kHz Neural Decoder...")
        t_dec0 = time.time()
        frame_tuples = [(gen_codes, None)]
        wav = codec.decode(frame_tuples).squeeze(0).squeeze(0).cpu().numpy()
        t_dec = time.time() - t_dec0

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    sf.write(output_path, wav, SAMPLE_RATE)
    audio_dur = len(wav) / SAMPLE_RATE

    print(f"🎉 Audio Successfully Synthesized to: {output_path}")
    print(f"📊 Audio Duration:  {audio_dur:.2f}s ({len(wav):,} samples @ {SAMPLE_RATE}Hz)")
    print(f"⏱️  Inference Time:  {(t_gen + t_dec)*1000:.2f} ms (Real-Time Factor: {(t_gen + t_dec)/audio_dur:.3f}x)")
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate drum sample waveform from text tags.")
    parser.add_argument("--prompt", type=str, default="808, kick, sub bass, deep, distorted, hard transient", help="Text tag prompt")
    parser.add_argument("--output", type=str, default="generated_drums/output_drum.wav", help="Output .wav path")
    parser.add_argument("--ckpt", type=str, default="checkpoints/drum_tgssm_best.pt", help="Checkpoint path")
    parser.add_argument("--temp", type=float, default=0.75, help="Sampling temperature")
    args = parser.parse_args()

    generate_sample_from_prompt(
        prompt=args.prompt,
        output_path=args.output,
        checkpoint_path=args.ckpt,
        temperature=args.temp,
    )
