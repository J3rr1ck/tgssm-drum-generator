"""
multi_genre_drum_dataset.py
Diverse Multi-Genre Drum Dataset Loader for 44.1kHz Continuous Flow-Matching TG-SSM.
Combines multiple real studio datasets from Hugging Face with robust error handling:
- yojul/one-shot-hip-hop-drums (Trap, Drill, Boom Bap, 808s)
- airasoul/drum-kit (Acoustic Rock, Indie, Funk Kits)
- DSP Analog Synth Physical Models (808, 909, LinnDrum, Synthwave, Latin Percussion)
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

from drum_synth_engine import generate_random_drum_sample

TARGET_SR = 44100
FIXED_SAMPLES = 22050 # 0.50 seconds @ 44.1kHz (43 DAC continuous frames)

GENRES = [
    "trap", "drill", "boom bap", "hip-hop", "synthwave", "80s retro",
    "techno", "house", "acoustic rock", "lo-fi jazz", "funk", "afrobeat", "latin percussion"
]

def analyze_acoustic_attributes(wav: np.ndarray, sr: int, genre_hint: str, label_hint: str) -> str:
    tags = []
    
    # 1. Add genre and type tags
    tags.append(genre_hint)
    
    label_norm = label_hint.lower().replace("_", " ").replace("-", " ")
    tags.append(label_norm)

    # 2. Spectral Analysis (Brightness & Sub-Bass)
    fft = np.abs(np.fft.rfft(wav))
    freqs = np.fft.rfftfreq(len(wav), 1.0 / sr)
    spectral_centroid = np.sum(freqs * fft) / (np.sum(fft) + 1e-8)

    if spectral_centroid > 6000:
        tags.append("bright top end")
        tags.append("crisp sheen")
    elif spectral_centroid < 900:
        tags.append("deep 50Hz sub")
        tags.append("heavy low end")
    elif spectral_centroid < 2200:
        tags.append("warm analog body")
        tags.append("punchy mid")
    else:
        tags.append("balanced presence")

    # 3. Transient Sharpness & Decay Time
    env = np.abs(wav)
    peak_idx = np.argmax(env)
    attack_time = peak_idx / sr
    
    if attack_time < 0.003:
        tags.append("hard transient click")
        tags.append("sharp attack")
    else:
        tags.append("soft envelope")

    peak_val = env[peak_idx] + 1e-6
    decay_indices = np.where(env[peak_idx:] < 0.1 * peak_val)[0]
    decay_time = (decay_indices[0] / sr) if len(decay_indices) > 0 else (len(wav) - peak_idx) / sr

    if decay_time < 0.10:
        tags.append("fast decay")
        tags.append("tight gated")
    elif decay_time > 0.35:
        tags.append("long decay")
        tags.append("diffused room reverb")
    else:
        tags.append("punchy sustain")

    if "808" in label_norm or "kick" in label_norm:
        sub_mask = (freqs >= 20) & (freqs <= 80)
        sub_energy = np.sum(fft[sub_mask] ** 2) / (np.sum(fft ** 2) + 1e-8)
        if sub_energy > 0.4:
            tags.append("resonant sub bass")
            tags.append("clean sine sweep")

    # Deduplicate while preserving order
    seen = set()
    unique = [t for t in tags if not (t in seen or seen.add(t))]
    return ", ".join(unique)

class MultiGenreDrumDataset(Dataset):
    def __init__(
        self,
        max_samples: int = 3000,
        cache_dir: str = "multi_genre_cache",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.max_samples = max_samples
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, f"multi_genre_drums_{max_samples}.pt")

        if os.path.exists(self.cache_file):
            print(f"Loading cached multi-genre 44.1kHz drum latents from {self.cache_file}...")
            cached = torch.load(self.cache_file, weights_only=False)
            self.entries = cached["entries"]
            print(f"✅ Loaded {len(self.entries):,} multi-genre continuous drum latents!")
        else:
            print(f"Aggregating and encoding {max_samples} multi-genre drum samples with DAC 44.1kHz...")
            self.entries = self._build_cache()
            torch.save({"entries": self.entries}, self.cache_file)
            print(f"✅ Saved multi-genre continuous latent cache ({len(self.entries)} samples)!")

    def _build_cache(self):
        # 1. Load DAC 44.1kHz Encoder
        model_path = dac.utils.download(model_type="44khz")
        codec = dac.DAC.load(model_path).to(self.device)
        codec.eval()

        entries = []
        target_a = int(self.max_samples * 0.60)
        target_b = int(self.max_samples * 0.25)
        target_synth = self.max_samples - target_a - target_b

        with torch.no_grad():
            # 2. Source A: yojul/one-shot-hip-hop-drums (Trap, Drill, Boom Bap, 808s)
            print("Ingesting Source A: yojul/one-shot-hip-hop-drums...")
            try:
                ds_a = load_dataset("yojul/one-shot-hip-hop-drums", split="train").cast_column("audio", Audio(decode=False))
                labels_a = ds_a.features["label"].names
                indices_a = random.sample(range(len(ds_a)), min(target_a * 2, len(ds_a)))
                
                count_a = 0
                for idx in indices_a:
                    if count_a >= target_a:
                        break
                    row = ds_a[idx]
                    label_name = labels_a[row["label"]]
                    genre = random.choice(["trap", "drill", "boom bap", "hip-hop", "lo-fi"])
                    
                    try:
                        if "audio" in row and "bytes" in row["audio"] and row["audio"]["bytes"]:
                            wav, sr = sf.read(io.BytesIO(row["audio"]["bytes"]))
                            entry = self._process_wav_sample(wav, sr, genre, label_name, codec)
                            if entry:
                                entries.append(entry)
                                count_a += 1
                    except Exception:
                        continue
                print(f"  ✅ Source A Ingested: {count_a} samples")
            except Exception as e:
                print(f"  ⚠️ Source A warning: {e}")

            # 3. Source B: airasoul/drum-kit (Acoustic Rock, Jazz, Funk Kits)
            print("Ingesting Source B: airasoul/drum-kit...")
            try:
                ds_b = load_dataset("airasoul/drum-kit", split="train").cast_column("audio", Audio(decode=False))
                indices_b = random.sample(range(len(ds_b)), min(target_b * 2, len(ds_b)))
                
                count_b = 0
                for idx in indices_b:
                    if count_b >= target_b:
                        break
                    row = ds_b[idx]
                    label_str = str(row.get("label", "acoustic drum"))
                    genre = random.choice(["acoustic rock", "indie jazz", "funk", "live studio"])
                    
                    try:
                        if "audio" in row and "bytes" in row["audio"] and row["audio"]["bytes"]:
                            wav, sr = sf.read(io.BytesIO(row["audio"]["bytes"]))
                            entry = self._process_wav_sample(wav, sr, genre, label_str, codec)
                            if entry:
                                entries.append(entry)
                                count_b += 1
                    except Exception:
                        continue
                print(f"  ✅ Source B Ingested: {count_b} samples")
            except Exception as e:
                print(f"  ⚠️ Source B warning: {e}")

            # 4. Source C (Synthesized Analog Models: Synthwave, 80s Retro, Latin Percussion, Afrobeat)
            needed_synth = self.max_samples - len(entries)
            print(f"Ingesting Source C (Procedural Analog Synth & Latin/Afrobeat Models): {needed_synth} samples...")
            for i in range(needed_synth):
                wav, synth_prompt = generate_random_drum_sample(24000)
                genre = random.choice(["synthwave", "80s retro", "techno", "house", "latin percussion", "afrobeat"])
                entry = self._process_wav_sample(wav, 24000, genre, synth_prompt, codec)
                if entry:
                    entries.append(entry)

        random.shuffle(entries)
        return entries

    def _process_wav_sample(self, wav: np.ndarray, sr: int, genre: str, label_hint: str, codec) -> dict:
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

        prompt_text = analyze_acoustic_attributes(wav, TARGET_SR, genre, label_hint)
        prompt_tokens = self.tokenizer.encode(
            prompt_text,
            max_length=28,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).squeeze(0)

        x = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0).to(self.device)
        x = codec.preprocess(x, TARGET_SR)
        z, _, _, _, _ = codec.encode(x)
        z_continuous = z.squeeze(0).cpu().to(torch.float32) # [1024, Frames]

        return {
            "prompt_ids": prompt_tokens,
            "continuous_z": z_continuous,
            "prompt_text": prompt_text,
            "genre": genre,
        }

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        return e["prompt_ids"], e["continuous_z"], e["prompt_text"]
