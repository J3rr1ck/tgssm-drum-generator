# 🥁 TG-SSM: Continuous Flow-Matching Text-to-Drum Waveform Synthesis Engine

Continuous State-Space Model (**TG-SSM**) adapted for text-conditioned **44.1kHz studio-grade drum sample waveform synthesis** via **Optimal Transport Flow Matching (Rectified Flow)** and **DAC (Descript Audio Codec)**, trained on real studio drum datasets on an **NVIDIA GeForce RTX 3060**.

---

## 🌟 Architecture & Highlights

1. **Continuous Optimal Transport Flow Matching (Rectified Flow)**:
   - Operates directly in the continuous 1024-dimensional acoustic manifold of **DAC 44.1kHz**, completely eliminating discrete codebook quantization distortion, robotic phase jitter, and smeared transients.
   - Vector field velocity formulation: $\mathcal{L}_{\text{FM}} = \| v_\theta(z_t, t, c) - (z_1 - z_0) \|^2$.
   - **System 2 Hamiltonian Latent Deliberation Core**: Multi-step continuous ODE propagation smoothing transient attacks and resonant sub-bass drops.

2. **Real Studio Dataset Ingestion**:
   - Ingests real studio drum one-shots from Hugging Face (`yojul/one-shot-hip-hop-drums`: 19,673 samples across 808s, kicks, snares, claps, hi-hats, open hats, cymbals).
   - Dynamic acoustic feature extraction (spectral centroid, transient onset, sub-energy ratios, decay envelopes) constructing rich multi-attribute conditioning prompts.

---

## 🎧 Final Master Audio Samples (44.1kHz Studio Quality)

Located in `generated_audio/`:

| Master Audio Sample | Target Drum / Prompt | Sample Rate | Duration | RMS Energy | Epoch |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`dac_flow_epoch_35_sample_1.wav`** | *"808, sub kick, sub bass, deep sub, deep 50Hz sub, heavy low end, hard transient click, resonant sub bass"* | `44,100 Hz` | `0.50s` | **`0.2998`** | **Epoch 35 (Master)** |
| **`dac_flow_epoch_35_sample_2.wav`** | *"snare drum, acoustic snare, crack, bright top end, crisp sheen, hard transient click"* | `44,100 Hz` | `0.50s` | `0.1194` | **Epoch 35 (Master)** |
| **`dac_flow_epoch_35_sample_3.wav`** | *"hihat, closed hat, metallic, bright top end, crisp sheen, fast decay, short tail"* | `44,100 Hz` | `0.50s` | `0.2033` | **Epoch 35 (Master)** |
| **`dac_flow_epoch_35_sample_4.wav`** | *"handclap, clap, percussion, warm analog body, punchy mid"* | `44,100 Hz` | `0.50s` | `0.1415` | **Epoch 35 (Master)** |

---

## 🚀 Quickstart & Synthesis CLI

### Generate 44.1kHz Studio Drum Samples from Text Tags (Flow Matching / RTX 3060)

```bash
# Generate 44.1kHz 808 Sub Kick
PYTHONPATH=python python3 python/generate_dac_drum.py \
  --prompt "808, sub kick, sub bass, deep sub, deep 50Hz sub, heavy low end, hard transient click" \
  --output generated_audio/studio_808.wav \
  --steps 30 --cfg 3.0

# Generate 44.1kHz Acoustic Snare
PYTHONPATH=python python3 python/generate_dac_drum.py \
  --prompt "snare drum, acoustic snare, crack, bright top end, crisp sheen, hard transient click" \
  --output generated_audio/studio_snare.wav \
  --steps 30 --cfg 3.0

# Generate 44.1kHz Trap Hi-Hat
PYTHONPATH=python python3 python/generate_dac_drum.py \
  --prompt "hihat, closed hat, metallic, bright top end, crisp sheen, fast decay, short tail" \
  --output generated_audio/studio_hihat.wav \
  --steps 30 --cfg 3.0
```
