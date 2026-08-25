"""
drum_synth_engine.py
Parametric DSP Drum Synthesizer Engine for studio-grade drum sample waveform generation.
Generates kicks, snares, hi-hats, claps, toms, and percussion with physical acoustic modeling
and rich descriptive prompt tags.
"""

import numpy as np
import scipy.signal
import soundfile as sf
import random
import os
from typing import Tuple, List, Dict

SAMPLE_RATE = 24000  # Matches Encodec 24kHz standard

def apply_soft_clip(x: np.ndarray, drive: float = 1.0) -> np.ndarray:
    """Analog-style hyperbolic tangent soft saturation."""
    if drive <= 1.0:
        return np.clip(x, -1.0, 1.0)
    return np.tanh(drive * x) / np.tanh(drive)

def synthesize_kick(
    duration: float = 0.6,
    f_start: float = 180.0,
    f_end: float = 45.0,
    pitch_decay: float = 0.04,
    amp_decay: float = 0.35,
    click_amount: float = 0.4,
    drive: float = 1.5,
    sr: int = SAMPLE_RATE,
) -> Tuple[np.ndarray, str]:
    """Synthesize 808 / 909 / Punchy Kick Drum."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Exponential pitch sweep
    pitch_env = f_end + (f_start - f_end) * np.exp(-t / pitch_decay)
    phase = 2 * np.pi * np.cumsum(pitch_env) / sr
    body = np.sin(phase)
    
    # Amplitude envelope
    amp_env = np.exp(-t / amp_decay)
    # Add punchy attack slope
    attack_samples = int(0.003 * sr)
    if attack_samples > 0:
        attack_curve = np.linspace(0, 1, attack_samples)
        amp_env[:attack_samples] *= attack_curve
        
    signal = body * amp_env
    
    # Attack transient click (filtered impulse/noise burst)
    click = np.random.uniform(-1, 1, len(t)) * np.exp(-t / 0.005)
    b, a = scipy.signal.butter(2, [1000 / (sr / 2), 6000 / (sr / 2)], btype='band')
    filtered_click = scipy.signal.lfilter(b, a, click)
    
    signal = signal + click_amount * filtered_click
    signal = apply_soft_clip(signal, drive)
    signal = signal / (np.max(np.abs(signal)) + 1e-6) * 0.95
    
    # Tag generation
    tags = ["kick", "drum"]
    if f_end < 50:
        tags.append("808")
        tags.append("sub bass")
        tags.append("deep")
    else:
        tags.append("punchy")
        tags.append("acoustic punch")
    if drive > 1.8:
        tags.append("distorted")
        tags.append("saturated")
    else:
        tags.append("clean")
    if click_amount > 0.4:
        tags.append("hard transient")
        tags.append("clicky")
    else:
        tags.append("soft attack")
        
    prompt = ", ".join(tags)
    return signal.astype(np.float32), prompt

def synthesize_snare(
    duration: float = 0.45,
    tone_freq1: float = 185.0,
    tone_freq2: float = 330.0,
    tone_decay: float = 0.08,
    noise_decay: float = 0.22,
    snare_brightness: float = 4500.0,
    noise_ratio: float = 0.65,
    drive: float = 1.3,
    sr: int = SAMPLE_RATE,
) -> Tuple[np.ndarray, str]:
    """Synthesize Acoustic / Electronic / Trap Snare Drum."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Dual tonal body oscillators
    body1 = np.sin(2 * np.pi * tone_freq1 * t) * np.exp(-t / tone_decay)
    body2 = np.sin(2 * np.pi * tone_freq2 * t) * np.exp(-t / (tone_decay * 0.8))
    body = 0.6 * body1 + 0.4 * body2
    
    # Snappy wire rattle noise
    noise = np.random.uniform(-1, 1, len(t))
    cutoff_norm = min(snare_brightness / (sr / 2), 0.95)
    b, a = scipy.signal.butter(2, [800 / (sr / 2), cutoff_norm], btype='band')
    filtered_noise = scipy.signal.lfilter(b, a, noise) * np.exp(-t / noise_decay)
    
    # Transient stick impact
    stick = np.random.uniform(-1, 1, len(t)) * np.exp(-t / 0.004)
    
    signal = (1.0 - noise_ratio) * body + noise_ratio * filtered_noise + 0.3 * stick
    signal = apply_soft_clip(signal, drive)
    signal = signal / (np.max(np.abs(signal)) + 1e-6) * 0.95
    
    tags = ["snare", "drum"]
    if snare_brightness > 5000:
        tags.append("bright")
        tags.append("crisp")
    else:
        tags.append("warm")
        tags.append("vintage")
    if noise_ratio > 0.7:
        tags.append("snappy wire rattle")
        tags.append("trap snare")
    else:
        tags.append("punchy body")
        tags.append("acoustic wood")
    if drive > 1.5:
        tags.append("analog crunch")
    else:
        tags.append("tight transient")
        
    prompt = ", ".join(tags)
    return signal.astype(np.float32), prompt

def synthesize_hihat(
    duration: float = 0.3,
    is_open: bool = False,
    highpass_freq: float = 7500.0,
    decay: float = 0.04,
    metallic_spread: float = 1.0,
    sr: int = SAMPLE_RATE,
) -> Tuple[np.ndarray, str]:
    """Synthesize 808/909 metallic pulse cluster Hi-Hat."""
    if is_open:
        duration = 0.65
        decay = 0.35
        
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # 6 Inharmonic square oscillators for authentic metallic sheen (TR-808/909 circuit)
    base_freqs = [205.3, 304.4, 369.6, 522.7, 540.0, 800.0]
    metallic = np.zeros_like(t)
    for f in base_freqs:
        freq = f * metallic_spread
        metallic += scipy.signal.square(2 * np.pi * freq * t)
    metallic /= len(base_freqs)
    
    # High-pass filter for crisp sizzle
    hp_norm = min(highpass_freq / (sr / 2), 0.95)
    b, a = scipy.signal.butter(3, hp_norm, btype='highpass')
    filtered = scipy.signal.lfilter(b, a, metallic)
    
    # Add subtle white noise sheen
    noise = np.random.uniform(-1, 1, len(t))
    b_n, a_n = scipy.signal.butter(2, 8000 / (sr / 2), btype='highpass')
    noise_filtered = scipy.signal.lfilter(b_n, a_n, noise)
    
    signal = 0.7 * filtered + 0.3 * noise_filtered
    amp_env = np.exp(-t / decay)
    signal = signal * amp_env
    signal = signal / (np.max(np.abs(signal)) + 1e-6) * 0.92
    
    tags = ["hihat", "cymbal"]
    if is_open:
        tags.append("open hat")
        tags.append("long decay")
        tags.append("sizzle")
    else:
        tags.append("closed hat")
        tags.append("fast decay")
        tags.append("tight tick")
    tags.append("metallic")
    tags.append("trap sizzle")
    tags.append("808 hat")
    
    prompt = ", ".join(tags)
    return signal.astype(np.float32), prompt

def synthesize_clap(
    duration: float = 0.5,
    num_bursts: int = 4,
    burst_spacing: float = 0.012,
    decay: float = 0.2,
    brightness: float = 3500.0,
    sr: int = SAMPLE_RATE,
) -> Tuple[np.ndarray, str]:
    """Synthesize Handclap with humanized micro-bursts."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    noise = np.random.uniform(-1, 1, len(t))
    
    # Bandpass filter for handclap resonance
    bp_low = max(600 / (sr / 2), 0.01)
    bp_high = min(brightness / (sr / 2), 0.95)
    b, a = scipy.signal.butter(2, [bp_low, bp_high], btype='band')
    filtered_noise = scipy.signal.lfilter(b, a, noise)
    
    env = np.zeros_like(t)
    for i in range(num_bursts):
        offset = int(i * burst_spacing * sr)
        if offset < len(t):
            burst_len = len(t) - offset
            burst_t = t[:burst_len]
            burst_env = np.exp(-burst_t / 0.01)
            env[offset:] += 0.7 * burst_env
            
    tail_offset = int(num_bursts * burst_spacing * sr)
    if tail_offset < len(t):
        tail_len = len(t) - tail_offset
        tail_t = t[:tail_len]
        tail_env = np.exp(-tail_t / decay)
        env[tail_offset:] += 1.0 * tail_env
        
    signal = filtered_noise * env
    signal = apply_soft_clip(signal, 1.4)
    signal = signal / (np.max(np.abs(signal)) + 1e-6) * 0.95
    
    tags = ["clap", "handclap", "percussion", "layered", "stereo snap", "reverb tail", "room resonance"]
    prompt = ", ".join(tags)
    return signal.astype(np.float32), prompt

def synthesize_tom(
    duration: float = 0.5,
    f_start: float = 240.0,
    f_end: float = 85.0,
    pitch_decay: float = 0.06,
    amp_decay: float = 0.28,
    sr: int = SAMPLE_RATE,
) -> Tuple[np.ndarray, str]:
    """Synthesize Electronic / Acoustic Tom Drum."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    pitch_env = f_end + (f_start - f_end) * np.exp(-t / pitch_decay)
    phase = 2 * np.pi * np.cumsum(pitch_env) / sr
    body = np.sin(phase) + 0.15 * np.sin(2 * phase)
    amp_env = np.exp(-t / amp_decay)
    signal = body * amp_env
    signal = apply_soft_clip(signal, 1.2)
    signal = signal / (np.max(np.abs(signal)) + 1e-6) * 0.95
    
    tags = ["tom", "drum", "percussion"]
    if f_end < 90:
        tags.append("low tom")
        tags.append("deep boom")
    elif f_end < 140:
        tags.append("mid tom")
        tags.append("punchy")
    else:
        tags.append("high tom")
        tags.append("resonant")
    tags.append("pitch sweep")
    
    prompt = ", ".join(tags)
    return signal.astype(np.float32), prompt

def generate_random_drum_sample(sr: int = SAMPLE_RATE) -> Tuple[np.ndarray, str]:
    """Generate a randomized drum sample from across all categories with variable parameters."""
    choice = random.choices(
        ["kick", "snare", "hihat_closed", "hihat_open", "clap", "tom"],
        weights=[0.30, 0.25, 0.15, 0.10, 0.10, 0.10],
        k=1
    )[0]
    
    if choice == "kick":
        f_start = random.uniform(120, 260)
        f_end = random.uniform(35, 65)
        pitch_decay = random.uniform(0.02, 0.07)
        amp_decay = random.uniform(0.18, 0.45)
        click_amount = random.uniform(0.1, 0.6)
        drive = random.uniform(1.0, 2.5)
        return synthesize_kick(0.6, f_start, f_end, pitch_decay, amp_decay, click_amount, drive, sr)
        
    elif choice == "snare":
        t1 = random.uniform(150, 220)
        t2 = random.uniform(280, 380)
        t_decay = random.uniform(0.05, 0.12)
        n_decay = random.uniform(0.12, 0.28)
        bright = random.uniform(3000, 7500)
        n_ratio = random.uniform(0.5, 0.85)
        drive = random.uniform(1.0, 2.0)
        return synthesize_snare(0.45, t1, t2, t_decay, n_decay, bright, n_ratio, drive, sr)
        
    elif choice == "hihat_closed":
        hp = random.uniform(6500, 9500)
        decay = random.uniform(0.02, 0.08)
        spread = random.uniform(0.8, 1.3)
        return synthesize_hihat(0.25, False, hp, decay, spread, sr)
        
    elif choice == "hihat_open":
        hp = random.uniform(6000, 8500)
        decay = random.uniform(0.25, 0.5)
        spread = random.uniform(0.85, 1.25)
        return synthesize_hihat(0.6, True, hp, decay, spread, sr)
        
    elif choice == "clap":
        bursts = random.randint(3, 5)
        spacing = random.uniform(0.008, 0.018)
        decay = random.uniform(0.15, 0.3)
        bright = random.uniform(2800, 4800)
        return synthesize_clap(0.5, bursts, spacing, decay, bright, sr)
        
    else:  # tom
        f_start = random.uniform(160, 320)
        f_end = random.uniform(60, 160)
        pitch_decay = random.uniform(0.04, 0.09)
        amp_decay = random.uniform(0.2, 0.4)
        return synthesize_tom(0.5, f_start, f_end, pitch_decay, amp_decay, sr)

if __name__ == "__main__":
    os.makedirs("test_drums", exist_ok=True)
    print("Testing DSP drum synthesis engine...")
    for i in range(10):
        wav, tags = generate_random_drum_sample()
        path = f"test_drums/drum_sample_{i+1}.wav"
        sf.write(path, wav, SAMPLE_RATE)
        print(f"Generated [{i+1:02d}]: {tags} -> {path} ({len(wav)} samples, {len(wav)/SAMPLE_RATE:.2f}s)")
