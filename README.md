# 🥁 TG-SSM: Continuous Flow-Matching Multi-Genre Drum Waveform Synthesis Engine

Continuous State-Space Model (**TG-SSM**) adapted for text-conditioned **44.1kHz studio-grade multi-genre drum sample waveform synthesis** via **Optimal Transport Flow Matching (Rectified Flow)** and **DAC (Descript Audio Codec)**.

Supported and trained on local **NVIDIA GeForce RTX 3060** & **Google AI Ultra / Colab Pro (A100, H100, L4)**.

---

## 🌟 Multi-Genre Architecture & High-Order Samplers

1. **Continuous Optimal Transport Flow Matching (Rectified Flow)**:
   - Operates directly in the continuous 1024-dimensional acoustic manifold of **DAC 44.1kHz**, completely eliminating discrete codebook quantization distortion, robotic phase jitter, and smeared transients.
   - Vector field velocity formulation: $\mathcal{L}_{\text{FM}} = \| v_\theta(z_t, t, c) - (z_1 - z_0) \|^2$.

2. **High-Order ODE Solvers**:
   - **`Heun` (2nd-Order Predictor-Corrector)**: Measures velocity at both start and predicted endpoints, averaging curvature for crisp transient snaps and solid sub-bass tracking.
   - **`RK4` (4th-Order Classic Runge-Kutta)**: High-precision 4-stage numerical integration.
   - **`Midpoint` (2nd-Order RK2)**: Smooth midpoint trajectory solver.
   - **`Euler` (1st-Order)**: Fast linear flow solver for rapid drafts.

3. **Diverse Multi-Genre Dataset Mixer**:
   - Ingests and blends multiple real studio datasets:
     * **Trap, Drill, Boom Bap & 808s**: `yojul/one-shot-hip-hop-drums` (19,673 samples)
     * **Acoustic Rock, Indie Jazz & Funk Kits**: `airasoul/drum-kit` (2,700 live samples)
     * **Synthwave, 80s Retro, Techno & House**: Analog physical models (TR-808, TR-909, LinnDrum, CR-78)
     * **Latin Percussion & Afrobeat**: Cowbells, congas, claves, woodblocks, shakers, handclaps

---

## 🎧 Master Audio Previews (Organized by Sampler)

Located in `generated_audio/<sampler>/`:

### 🔬 1. Heun 2nd-Order Predictor-Corrector (`generated_audio/heun/`)

| Audio Sample | Target Drum / Conditioning Tag Prompt | Sample Rate | Duration | RMS Energy | Inference Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`master_heun_808.wav`** | *"trap, 808, sub kick, deep sub, deep 50Hz sub, heavy low end, hard transient click"* | `44,100 Hz` | `0.50s` | **`0.3559`** | `11.7s` |
| **`master_heun_snare.wav`** | *"acoustic rock, snare drum, live studio crack, bright top end, crisp sheen"* | `44,100 Hz` | `0.50s` | **`0.2015`** | `11.0s` |
| **`master_heun_synthwave.wav`** | *"synthwave, 80s retro, gated snare, analog saturation, diffused room reverb"* | `44,100 Hz` | `0.50s` | **`0.2198`** | `10.3s` |
| **`master_heun_cowbell.wav`** | *"latin percussion, afrobeat, cowbell, woodblock, bright top end, metallic"* | `44,100 Hz` | `0.50s` | **`0.2615`** | `10.7s` |

### ⚡ 2. RK4 4th-Order Classic Runge-Kutta (`generated_audio/rk4/`)

| Audio Sample | Target Drum / Conditioning Tag Prompt | Sample Rate | Duration | RMS Energy | Inference Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`master_rk4_808.wav`** | *"trap, 808, sub kick, deep sub, deep 50Hz sub, heavy low end, hard transient click"* | `44,100 Hz` | `0.50s` | **`0.4060`** | `21.1s` |
| **`master_rk4_snare.wav`** | *"acoustic rock, snare drum, live studio crack, bright top end, crisp sheen"* | `44,100 Hz` | `0.50s` | **`0.1621`** | `20.7s` |
| **`master_rk4_synthwave.wav`** | *"synthwave, 80s retro, gated snare, analog saturation, diffused room reverb"* | `44,100 Hz` | `0.50s` | **`0.2019`** | `21.1s` |
| **`master_rk4_cowbell.wav`** | *"latin percussion, afrobeat, cowbell, woodblock, bright top end, metallic"* | `44,100 Hz` | `0.50s` | **`0.1736`** | `20.8s` |

### 🌊 3. Midpoint 2nd-Order RK2 (`generated_audio/midpoint/`)

| Audio Sample | Target Drum / Conditioning Tag Prompt | Sample Rate | Duration | RMS Energy | Inference Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`master_midpoint_808.wav`** | *"trap, 808, sub kick, deep sub, deep 50Hz sub, heavy low end, hard transient click"* | `44,100 Hz` | `0.50s` | **`0.4195`** | `10.3s` |
| **`master_midpoint_snare.wav`** | *"acoustic rock, snare drum, live studio crack, bright top end, crisp sheen"* | `44,100 Hz` | `0.50s` | **`0.1522`** | `11.3s` |
| **`master_midpoint_synthwave.wav`** | *"synthwave, 80s retro, gated snare, analog saturation, diffused room reverb"* | `44,100 Hz` | `0.50s` | **`0.1520`** | `10.5s` |
| **`master_midpoint_cowbell.wav`** | *"latin percussion, afrobeat, cowbell, woodblock, bright top end, metallic"* | `44,100 Hz` | `0.50s` | **`0.1706`** | `10.5s` |

### 🚀 4. Euler 1st-Order Flow (`generated_audio/euler/`)

* Master Multi-Genre Samples: `master_euler_808.wav`, `master_euler_snare.wav`, `master_euler_synthwave.wav`, `master_euler_cowbell.wav` (`~5.0s` per sample).
* Training Checkpoint Audio Archive: `generated_audio/euler/training_checkpoints/` (Historical Epoch 5–35 milestone previews).

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
