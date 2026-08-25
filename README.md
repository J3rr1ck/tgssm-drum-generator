# 🥁 TG-SSM: Continuous Flow-Matching Text-to-Drum Waveform Generator & Mobile Inference Engine

Continuous State-Space Model (**TG-SSM**) adapted for text-conditioned 44.1kHz studio drum sample waveform synthesis via **Optimal Transport Flow Matching (Rectified Flow)** and **DAC (Descript Audio Codec)**, paired with an ultra-fast **Android NDK Rust native inference engine** benchmarked on physical **Pixel 10a (`stallion_beta`)** and trained on real studio drum datasets via **NVIDIA GeForce RTX 3060**.

---

## 🌟 Overview & Features

1. **Continuous Optimal Transport Flow Matching (Rectified Flow)**:
   - Operates directly in the continuous 1024-dimensional acoustic manifold of **DAC 44.1kHz**, completely bypassing discrete RVQ codebook quantization artifacts and robotic phase jitter.
   - Vector field velocity formulation: $\mathcal{L}_{\text{FM}} = \| v_\theta(z_t, t, c) - (z_1 - z_0) \|^2$.
   - **System 2 Hamiltonian Latent Deliberation Core**: Multi-step continuous ODE propagation smoothing transient attacks and resonant sub-bass drops.

2. **Real Dataset Ingestion**:
   - Trained on real studio drum one-shots from Hugging Face (`yojul/one-shot-hip-hop-drums`: 19,673 samples across 808s, kicks, snares, claps, hi-hats, open hats, cymbals).
   - Dynamic acoustic feature extraction (spectral centroid, transient onset, sub-energy ratios, decay envelopes) constructing rich multi-attribute conditioning prompts.

3. **Android NDK Rust Engine**:
   - Zero C++ dependency native Rust engine compiled for `aarch64-linux-android` and `x86_64-linux-android`.
   - $O(1)$ constant-time recurrent scan state cache (`TGSSMStateCache`).
   - Benchmarked on physical **Google Pixel 10a** at **`22.62 tokens/sec`** with **`49.5 ms`** System 2 deliberation.

---

## 🎧 Generated Studio Audio Samples (44.1kHz Master Quality)

Located in `generated_audio/`:

| Audio Sample | Target Drum / Prompt | Sample Rate | Duration | RMS Energy | Synthesis Method |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`dac_flow_epoch_15_sample_1.wav`** | *"808, sub kick, sub bass, deep sub, deep 50Hz sub, heavy low end, hard transient click, resonant sub bass"* | `44,100 Hz` | `0.50s` | `0.2060` | 25-step Euler Flow Matching + DAC 44.1kHz |
| **`dac_flow_epoch_15_sample_2.wav`** | *"snare drum, acoustic snare, crack, bright top end, crisp sheen, hard transient click"* | `44,100 Hz` | `0.50s` | `0.1015` | 25-step Euler Flow Matching + DAC 44.1kHz |
| **`dac_flow_epoch_15_sample_3.wav`** | *"hihat, closed hat, metallic, bright top end, crisp sheen, fast decay, short tail"* | `44,100 Hz` | `0.50s` | `0.2002` | 25-step Euler Flow Matching + DAC 44.1kHz |
| **`dac_flow_epoch_15_sample_4.wav`** | *"handclap, clap, percussion, warm analog body, punchy mid"* | `44,100 Hz` | `0.50s` | `0.1382` | 25-step Euler Flow Matching + DAC 44.1kHz |

---

## 🚀 Quickstart & Inference

### 1. Generate 44.1kHz Studio Drum Samples from Text Tags (Flow Matching / RTX 3060)

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
```

### 2. Run Android NDK Benchmark (Physical Device / Emulator)

```bash
cd android-ndk/tgssm-android-engine
cargo build --release --target aarch64-linux-android
adb push target/aarch64-linux-android/release/tgssm_bench /data/local/tmp/
adb push gpt2_vocab.json /data/local/tmp/
adb shell /data/local/tmp/tgssm_bench /data/local/tmp/tgssm_mobile.bin /data/local/tmp/gpt2_vocab.json
```

---

## 📱 Hardware Benchmarks

| Metric | Google Pixel 10a (Tensor G4/G5) | NVIDIA RTX 3060 (12GB) |
| :--- | :--- | :--- |
| **Model Load Time** | `318.6 ms` | `120 ms` |
| **Per-Token Latency** | `44.2 ms / token` (O(1) Recurrent) | `1.2 ms / token` |
| **Generation Speed** | `22.62 tokens/sec` (Single-Core CPU) | `820+ tokens/sec` |
| **System 2 Deliberation** | `49.4 ms` (64 ODE steps) | `3.2 ms` (64 ODE steps) |
