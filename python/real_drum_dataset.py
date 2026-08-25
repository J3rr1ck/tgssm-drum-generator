"""
real_drum_dataset.py
PyTorch Dataset for Real Studio Drum Samples from Hugging Face (yojul/one-shot-hip-hop-drums).
Extracts real audio waveforms, computes acoustic attributes for rich prompt tags,
and encodes audio via EnCodec 24kHz RVQ neural tokens.
"""

import io
import os
import random
import numpy as np
import soundfile as sf
import scipy.signal
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, Audio
from transformers import AutoTokenizer
import encodec

TARGET_SR = 24000
FIXED_FRAMES = 48  # Exactly 0.64s @ 75Hz (15,360 audio samples)

def compute_acoustic_tags(wav: np.ndarray, sr: int, base_label: str) -> str:
    """Extract acoustic descriptors from real drum waveform to build a rich descriptive prompt."""
    tags = []
    
    # Base label normalization
    label_map = {
        "808S": ["808", "sub kick", "sub bass", "deep"],
        "KICKS": ["kick", "drum", "punchy bass drum"],
        "SNARES": ["snare", "drum", "acoustic snare"],
        "CLAPS": ["clap", "handclap", "percussion"],
        "HITHATS": ["hihat", "closed hat", "metallic"],
        "OPENHATS": ["open hat", "cymbal", "sizzle"],
        "CYMBALS": ["cymbal", "crash", "ride"],
    }
    tags.extend(label_map.get(base_label, [base_label.lower(), "drum"]))

    # 1. Spectral Centroid (Brightness)
    fft = np.abs(np.fft.rfft(wav))
    freqs = np.fft.rfftfreq(len(wav), 1.0 / sr)
    spectral_centroid = np.sum(freqs * fft) / (np.sum(fft) + 1e-8)

    if spectral_centroid > 5000:
        tags.append("bright")
        tags.append("crisp")
    elif spectral_centroid < 800:
        tags.append("deep low end")
        tags.append("heavy sub")
    else:
        tags.append("warm")
        tags.append("punchy mid")

    # 2. Transient Sharpness & Decay Time
    env = np.abs(wav)
    peak_idx = np.argmax(env)
    attack_time = peak_idx / sr
    
    # Measure decay to 10% peak
    peak_val = env[peak_idx] + 1e-6
    decay_indices = np.where(env[peak_idx:] < 0.1 * peak_val)[0]
    decay_time = (decay_indices[0] / sr) if len(decay_indices) > 0 else (len(wav) - peak_idx) / sr

    if attack_time < 0.005:
        tags.append("hard transient")
        tags.append("tight attack")
    else:
        tags.append("soft attack")

    if decay_time < 0.12:
        tags.append("fast decay")
        tags.append("short tail")
    elif decay_time > 0.4:
        tags.append("long decay")
        tags.append("reverberant")
    else:
        tags.append("punchy body")

    if base_label in ["808S", "KICKS"]:
        # Check sub ratio (20-80 Hz)
        sub_mask = (freqs >= 20) & (freqs <= 90)
        sub_energy = np.sum(fft[sub_mask] ** 2) / (np.sum(fft ** 2) + 1e-8)
        if sub_energy > 0.4:
            tags.append("50Hz sub punch")
            tags.append("heavy bottom")

    # Deduplicate while preserving order
    seen = set()
    unique_tags = [t for t in tags if not (t in seen or seen.add(t))]
    return ", ".join(unique_tags)

class RealDrumDataset(Dataset):
    def __init__(
        self,
        max_samples: int = 4000,
        fixed_audio_samples: int = 15360, # 48 Encodec frames
        cache_dir: str = "real_drum_cache",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.max_samples = max_samples
        self.fixed_len = fixed_audio_samples
        self.device = device
        
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, f"real_drum_dataset_{max_samples}.pt")

        if os.path.exists(self.cache_file):
            print(f"Loading cached real drum dataset from {self.cache_file}...")
            cached = torch.load(self.cache_file, weights_only=False)
            self.entries = cached["entries"]
            print(f"✅ Loaded {len(self.entries):,} pre-encoded real drum samples!")
        else:
            print(f"Downloading & preparing real drum dataset from Hugging Face (yojul/one-shot-hip-hop-drums)...")
            self.entries = self._build_cache()
            print(f"Saving real drum cache to {self.cache_file}...")
            torch.save({"entries": self.entries}, self.cache_file)
            print(f"✅ Successfully cached {len(self.entries):,} real drum samples!")

    def _build_cache(self):
        hf_ds = load_dataset("yojul/one-shot-hip-hop-drums", split="train")
        hf_ds = hf_ds.cast_column("audio", Audio(decode=False))
        labels = hf_ds.features["label"].names

        codec = encodec.EncodecModel.encodec_model_24khz().to(self.device)
        codec.set_target_bandwidth(6.0)
        codec.eval()

        total = min(self.max_samples, len(hf_ds))
        indices = list(range(len(hf_ds)))
        random.seed(42)
        random.shuffle(indices)
        selected_indices = indices[:total]

        entries = []
        with torch.no_grad():
            for count, idx in enumerate(selected_indices):
                row = hf_ds[idx]
                audio_bytes = row["audio"]["bytes"]
                base_label = labels[row["label"]]

                # Read audio
                wav, sr = sf.read(io.BytesIO(audio_bytes))
                
                # Convert stereo to mono
                if len(wav.shape) > 1 and wav.shape[1] > 1:
                    wav = np.mean(wav, axis=1)

                # Resample to 24,000 Hz if needed
                if sr != TARGET_SR:
                    num_target = int(len(wav) * TARGET_SR / sr)
                    wav = scipy.signal.resample(wav, num_target)

                # Normalize and crop/pad to fixed length (15,360 samples = 48 frames)
                if len(wav) < self.fixed_len:
                    padded = np.zeros(self.fixed_len, dtype=np.float32)
                    padded[:len(wav)] = wav
                    wav = padded
                else:
                    wav = wav[:self.fixed_len].astype(np.float32)

                peak = np.max(np.abs(wav)) + 1e-6
                wav = (wav / peak) * 0.95

                # Compute rich tags
                prompt_text = compute_acoustic_tags(wav, TARGET_SR, base_label)

                # Tokenize prompt text
                prompt_tokens = self.tokenizer.encode(
                    prompt_text,
                    max_length=24,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).squeeze(0)

                # Encode to 8 EnCodec RVQ codebook indices
                wav_t = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0).to(self.device)
                frames = codec.encode(wav_t)
                codes = frames[0][0].squeeze(0).cpu().to(torch.long) # [8, 48]

                entries.append({
                    "prompt_ids": prompt_tokens,
                    "audio_codes": codes,
                    "prompt_text": prompt_text,
                    "base_label": base_label,
                })

                if (count + 1) % 500 == 0 or (count + 1) == total:
                    print(f"  Processed [{count+1}/{total}] real drum samples ({base_label})...")

        return entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        return e["prompt_ids"], e["audio_codes"], e["prompt_text"]

if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = RealDrumDataset(max_samples=50, device=dev)
    p_ids, codes, text = ds[0]
    print(f"\nReal Sample 0:")
    print(f"  Prompt: '{text}'")
    print(f"  Prompt IDs Shape: {p_ids.shape}")
    print(f"  Audio Codes Shape: {codes.shape} (8 codebooks, {codes.shape[1]} frames)")
