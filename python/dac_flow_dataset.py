"""
dac_flow_dataset.py
Continuous Latent Dataset using DAC (Descript Audio Codec 44.1kHz).
Extracts 1024-dimensional continuous latent trajectories z from real studio drum samples
and pairs them with descriptive text tags for Flow Matching.
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
import dac

TARGET_SR = 44100
FIXED_SAMPLES = 22050 # 0.50 seconds @ 44.1kHz = exactly 43 DAC latent frames (hop size 512)

def compute_rich_drum_prompt(wav: np.ndarray, sr: int, base_label: str) -> str:
    tags = []
    label_map = {
        "808S": ["808", "sub kick", "sub bass", "deep sub"],
        "KICKS": ["kick drum", "punchy bass drum", "acoustic kick"],
        "SNARES": ["snare drum", "acoustic snare", "crack"],
        "CLAPS": ["handclap", "clap", "percussion"],
        "HITHATS": ["hihat", "closed hat", "metallic"],
        "OPENHATS": ["open hat", "cymbal", "sizzle"],
        "CYMBALS": ["cymbal", "crash", "ride"],
    }
    tags.extend(label_map.get(base_label, [base_label.lower(), "drum"]))

    fft = np.abs(np.fft.rfft(wav))
    freqs = np.fft.rfftfreq(len(wav), 1.0 / sr)
    spectral_centroid = np.sum(freqs * fft) / (np.sum(fft) + 1e-8)

    if spectral_centroid > 6000:
        tags.append("bright top end")
        tags.append("crisp sheen")
    elif spectral_centroid < 1000:
        tags.append("deep 50Hz sub")
        tags.append("heavy low end")
    else:
        tags.append("warm analog body")
        tags.append("punchy mid")

    env = np.abs(wav)
    peak_idx = np.argmax(env)
    attack_time = peak_idx / sr
    
    if attack_time < 0.004:
        tags.append("hard transient click")
        tags.append("tight attack")
    else:
        tags.append("soft envelope")

    if base_label in ["808S", "KICKS"]:
        sub_mask = (freqs >= 25) & (freqs <= 85)
        sub_energy = np.sum(fft[sub_mask] ** 2) / (np.sum(fft ** 2) + 1e-8)
        if sub_energy > 0.4:
            tags.append("resonant sub bass")
            tags.append("clean sine drop")

    seen = set()
    unique = [t for t in tags if not (t in seen or seen.add(t))]
    return ", ".join(unique)

class DACContinuousDrumDataset(Dataset):
    def __init__(
        self,
        max_samples: int = 2000,
        cache_dir: str = "dac_drum_cache",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.max_samples = max_samples
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, f"dac_continuous_drums_{max_samples}.pt")

        if os.path.exists(self.cache_file):
            print(f"Loading cached DAC continuous latents from {self.cache_file}...")
            cached = torch.load(self.cache_file, weights_only=False)
            self.entries = cached["entries"]
            print(f"✅ Loaded {len(self.entries):,} continuous 44.1kHz drum latents!")
        else:
            print(f"Extracting DAC 44.1kHz continuous latents for {max_samples} real drum samples...")
            self.entries = self._build_cache()
            torch.save({"entries": self.entries}, self.cache_file)
            print(f"✅ Saved DAC continuous latent cache ({len(self.entries)} samples)!")

    def _build_cache(self):
        hf_ds = load_dataset("yojul/one-shot-hip-hop-drums", split="train")
        hf_ds = hf_ds.cast_column("audio", Audio(decode=False))
        labels = hf_ds.features["label"].names

        model_path = dac.utils.download(model_type="44khz")
        codec = dac.DAC.load(model_path).to(self.device)
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

                wav, sr = sf.read(io.BytesIO(audio_bytes))
                if len(wav.shape) > 1 and wav.shape[1] > 1:
                    wav = np.mean(wav, axis=1)

                if sr != TARGET_SR:
                    num_target = int(len(wav) * TARGET_SR / sr)
                    wav = scipy.signal.resample(wav, num_target)

                if len(wav) < FIXED_SAMPLES:
                    padded = np.zeros(FIXED_SAMPLES, dtype=np.float32)
                    padded[:len(wav)] = wav
                    wav = padded
                else:
                    wav = wav[:FIXED_SAMPLES].astype(np.float32)

                peak = np.max(np.abs(wav)) + 1e-6
                wav = (wav / peak) * 0.95

                prompt_text = compute_rich_drum_prompt(wav, TARGET_SR, base_label)
                prompt_tokens = self.tokenizer.encode(
                    prompt_text,
                    max_length=24,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).squeeze(0)

                x = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0).to(self.device)
                x = codec.preprocess(x, TARGET_SR)
                # z: [1, 1024, Frames] continuous latent representation
                z, _, _, _, _ = codec.encode(x)
                z_continuous = z.squeeze(0).cpu().to(torch.float32) # [1024, Frames]

                entries.append({
                    "prompt_ids": prompt_tokens,
                    "continuous_z": z_continuous,
                    "prompt_text": prompt_text,
                })

                if (count + 1) % 400 == 0 or (count + 1) == total:
                    print(f"  Processed [{count+1}/{total}] real drum samples ({base_label})...")

        return entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        return e["prompt_ids"], e["continuous_z"], e["prompt_text"]
