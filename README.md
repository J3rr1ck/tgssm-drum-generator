# 🥁 TG-SSM: Continuous Flow-Matching Multi-Genre Drum Waveform Synthesis Engine

Continuous State-Space Model (**TG-SSM**) adapted for text-conditioned **44.1kHz studio-grade multi-genre drum sample waveform synthesis** via **Optimal Transport Flow Matching (Rectified Flow)** and **DAC (Descript Audio Codec)**.

Supported and trained on local **NVIDIA GeForce RTX 3060** & **Google AI Ultra / Colab Pro (A100, H100, L4)**.

---

## 🌟 Multi-Genre Architecture & High-Order Samplers

1. **Continuous Optimal Transport Flow Matching (Rectified Flow)**:
   - Operates directly in the continuous 1024-dimensional acoustic manifold of **DAC 44.1kHz**, completely eliminating discrete codebook quantization distortion, robotic phase jitter, and smeared transients.
   - Vector field velocity formulation: $\mathcal{L}_{\text{FM}} = \| v_\theta(z_t, t, c) - (z_1 - z_0) \|^2$.

2. **High-Order ODE Solvers**:
   - **`Heun` (2nd-Order Predictor-Corrector)**: Accurately tracks rapid curvature in transient bursts and pitch drops, yielding punchier transients and cleaner sub bass.
   - **`RK4` (4th-Order Classic Runge-Kutta)**: High-precision 4th-order multi-stage integration.
   - **`Midpoint` (2nd-Order RK2)**: Balanced harmonic trajectory solver.
   - **`Euler` (1st-Order)**: Fast linear flow solver for rapid drafts.

3. **Diverse Multi-Genre Dataset Mixer**:
   - Ingests and blends multiple real studio datasets:
     * **Trap, Drill, Boom Bap & 808s**: `yojul/one-shot-hip-hop-drums` (19,673 samples)
     * **Acoustic Rock, Indie Jazz & Funk Kits**: `airasoul/drum-kit` (2,700 live samples)
     * **Synthwave, 80s Retro, Techno & House**: Analog physical models (TR-808, TR-909, LinnDrum, CR-78)
     * **Latin Percussion & Afrobeat**: Cowbells, congas, claves, woodblocks, shakers, handclaps

---

## 🎧 Generated Studio Audio Samples (Organized by Sampler)

Located in `generated_audio/<sampler>/`:

### 🔬 1. Heun 2nd-Order Predictor-Corrector (`generated_audio/heun/`)

| Audio Sample | Target Drum / Conditioning Prompt | Sample Rate | Duration | RMS Energy | Sampler |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`test_heun_808.wav`** | *"trap, 808, sub kick, deep sub, deep 50Hz sub, heavy low end, hard transient click"* | `44,100 Hz` | `0.50s` | `0.2111` | **Heun (2nd-Order)** |
| **`test_heun_snare.wav`** | *"acoustic rock, snare drum, live studio crack, bright top end, crisp sheen"* | `44,100 Hz` | `0.50s` | `0.1527` | **Heun (2nd-Order)** |

### ⚡ 2. Euler 1st-Order Flow (`generated_audio/euler/`)

| Audio Sample | Target Drum / Conditioning Prompt | Sample Rate | Duration | RMS Energy | Epoch |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`dac_flow_epoch_35_sample_1.wav`** | *"808, sub kick, sub bass, deep sub, deep 50Hz sub, heavy low end, hard transient click"* | `44,100 Hz` | `0.50s` | **`0.2998`** | **Master (Epoch 35)** |
| **`dac_flow_epoch_35_sample_2.wav`** | *"snare drum, acoustic snare, crack, bright top end, crisp sheen, hard transient click"* | `44,100 Hz` | `0.50s` | `0.1194` | **Master (Epoch 35)** |
| **`multi_genre_epoch_10_sample_1.wav`** | *"trap, 808, sub kick, deep sub, deep 50Hz sub, heavy low end, hard transient click"* | `44,100 Hz` | `0.50s` | `0.2218` | **Multi-Genre (Epoch 10)** |
| **`multi_genre_epoch_10_sample_2.wav`** | *"acoustic rock, snare drum, live studio crack, bright top end, crisp sheen"* | `44,100 Hz` | `0.50s` | `0.2249` | **Multi-Genre (Epoch 10)** |

---

## 🚀 CLI Synthesis with Sampler Selection

```bash
# 1. Generate 808 Sub Kick using HEUN (2nd-Order Predictor-Corrector)
PYTHONPATH=python python3 python/generate_dac_drum.py \
  --prompt "trap, 808, sub kick, deep sub, deep 50Hz sub, heavy low end, hard transient click" \
  --output generated_audio/heun/my_808.wav \
  --sampler heun --steps 30 --cfg 3.5

# 2. Generate Acoustic Snare using RK4 (4th-Order)
PYTHONPATH=python python3 python/generate_dac_drum.py \
  --prompt "acoustic rock, snare drum, live studio crack, bright top end, crisp sheen" \
  --output generated_audio/rk4/my_snare.wav \
  --sampler rk4 --steps 25 --cfg 3.5

# 3. Generate Synthwave Snare using MIDPOINT
PYTHONPATH=python python3 python/generate_dac_drum.py \
  --prompt "synthwave, 80s retro, gated snare, analog saturation, diffused room reverb" \
  --output generated_audio/midpoint/my_snare.wav \
  --sampler midpoint --steps 30 --cfg 3.0
```
