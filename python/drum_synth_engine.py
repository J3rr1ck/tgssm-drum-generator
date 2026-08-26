"""
drum_synth_engine.py
Upgraded DSP Drum Synthesizer Engine with Authentic Roland TR-808 Cowbell, Latin Agogo, Timbales, and High-Definition Percussion.
"""

import numpy as np
import scipy.signal
import random

def generate_authentic_808_cowbell(sr: int = 44100, dur: float = 0.5) -> np.ndarray:
    """
    Authentic Roland TR-808 Cowbell Circuit Emulation:
    - Dual square wave oscillators at 540 Hz and 800 Hz
    - High-Q bandpass filtering (Q ~ 4.0) with resonant peak
    - Fast transient strike with dual-decay metallic envelope
    """
    n_samples = int(sr * dur)
    t = np.linspace(0, dur, n_samples, endpoint=False)

    # 1. Dual Square Wave Oscillators (Classic 808 frequencies)
    f1 = 540.0 + random.uniform(-15.0, 15.0)
    f2 = 800.0 + random.uniform(-20.0, 20.0)
    
    sq1 = scipy.signal.square(2 * np.pi * f1 * t)
    sq2 = scipy.signal.square(2 * np.pi * f2 * t)
    raw = 0.5 * sq1 + 0.5 * sq2

    # Add metallic overtone harmonics (inharmonic ring)
    f3 = 1.48 * f2
    f4 = 2.65 * f2
    ring = 0.3 * np.sin(2 * np.pi * f3 * t) + 0.2 * np.sin(2 * np.pi * f4 * t)
    raw = raw + ring

    # 2. Dual-stage Exponential Envelope
    # Strike click (instant transient) + metallic body ring
    strike_env = np.exp(-t / 0.015)
    body_env = np.exp(-t / 0.12)
    env = 0.6 * strike_env + 0.4 * body_env

    # 3. High-Q Bandpass Filter centered at 800 Hz
    sos = scipy.signal.butter(2, [600 / (sr / 2), 1600 / (sr / 2)], btype='bandpass', output='sos')
    filtered = scipy.signal.sosfilt(sos, raw * env)

    # 4. Add transient strike noise click
    click = np.random.randn(n_samples) * np.exp(-t / 0.003) * 0.4
    out = filtered + click

    # Peak normalization
    peak = np.max(np.abs(out)) + 1e-6
    return (out / peak * 0.95).astype(np.float32)

def generate_latin_agogo_bell(sr: int = 44100, dur: float = 0.5, high_pitch: bool = False) -> np.ndarray:
    """Latin Agogo / Samba Bell physical model."""
    n_samples = int(sr * dur)
    t = np.linspace(0, dur, n_samples, endpoint=False)
    
    base_f = 950.0 if high_pitch else 620.0
    f1 = base_f + random.uniform(-10, 10)
    f2 = base_f * 1.58
    f3 = base_f * 2.82
    
    bell = (
        0.5 * np.sin(2 * np.pi * f1 * t) +
        0.35 * np.sin(2 * np.pi * f2 * t) +
        0.15 * np.sin(2 * np.pi * f3 * t)
    )
    
    env = np.exp(-t / 0.18)
    click = np.random.randn(n_samples) * np.exp(-t / 0.002) * 0.3
    out = bell * env + click
    
    peak = np.max(np.abs(out)) + 1e-6
    return (out / peak * 0.95).astype(np.float32)

def generate_latin_timbale(sr: int = 44100, dur: float = 0.5) -> np.ndarray:
    """Latin Timbale with sharp rimshot transient and ringing metallic body."""
    n_samples = int(sr * dur)
    t = np.linspace(0, dur, n_samples, endpoint=False)
    
    f_start = 480.0
    f_end = 220.0
    freq_env = f_end + (f_start - f_end) * np.exp(-t / 0.02)
    phase = 2 * np.pi * np.cumsum(freq_env) / sr
    
    body = np.sin(phase) * np.exp(-t / 0.15)
    rim_ring = np.sin(2 * np.pi * 1850 * t) * np.exp(-t / 0.04) * 0.4
    rim_click = np.random.randn(n_samples) * np.exp(-t / 0.003) * 0.6
    
    out = body + rim_ring + rim_click
    peak = np.max(np.abs(out)) + 1e-6
    return (out / peak * 0.95).astype(np.float32)

def generate_afrobeat_shaker(sr: int = 44100, dur: float = 0.5) -> np.ndarray:
    """Afrobeat / Latin High-Definition Seed Shaker."""
    n_samples = int(sr * dur)
    t = np.linspace(0, dur, n_samples, endpoint=False)
    
    noise = np.random.randn(n_samples)
    sos = scipy.signal.butter(3, [3500 / (sr / 2), 16000 / (sr / 2)], btype='bandpass', output='sos')
    filtered = scipy.signal.sosfilt(sos, noise)
    
    # Forward and backward shaker envelope
    attack = np.exp(-((t - 0.03) ** 2) / (2 * (0.015 ** 2)))
    decay = np.exp(-t / 0.08)
    env = 0.7 * attack + 0.3 * decay
    
    out = filtered * env
    peak = np.max(np.abs(out)) + 1e-6
    return (out / peak * 0.95).astype(np.float32)

def generate_random_drum_sample(sr: int = 44100) -> tuple:
    """Randomly generate high-definition drum/percussion one-shots."""
    choice = random.choice([
        "cowbell", "agogo_high", "agogo_low", "timbale", "shaker",
        "808_cowbell", "latin_bell"
    ])
    
    if choice in ["cowbell", "808_cowbell"]:
        wav = generate_authentic_808_cowbell(sr)
        prompt = "808 cowbell, metallic cowbell, dual square wave, sharp transient click, crisp sheen, bright top end"
    elif choice == "agogo_high":
        wav = generate_latin_agogo_bell(sr, high_pitch=True)
        prompt = "latin agogo bell, high samba bell, metallic ring, bright top end, crisp sheen, hard transient"
    elif choice == "agogo_low":
        wav = generate_latin_agogo_bell(sr, high_pitch=False)
        prompt = "latin agogo bell, low samba bell, metallic ring, warm analog body, punchy mid"
    elif choice == "timbale":
        wav = generate_latin_timbale(sr)
        prompt = "latin timbale, timbales rimshot, metallic ring, hard transient click, punchy mid"
    else:
        wav = generate_afrobeat_shaker(sr)
        prompt = "afrobeat shaker, latin shaker, bright top end, crisp sheen, fast decay, tight gated"
        
    return wav, prompt
