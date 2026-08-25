# 🥁 TG-SSM: Text-to-Drum Waveform Synthesis & Mobile Inference Engine

Continuous State-Space Model (**TG-SSM**) adapted for text-conditioned drum sample waveform synthesis, paired with an ultra-fast **Android NDK Rust native inference engine** benchmarked on physical **Pixel 10a (`stallion_beta`)** and trained on real studio drum datasets via **NVIDIA GeForce RTX 3060**.

---

## 🌟 Overview & Features

1. **Continuous Selective SSM (S6) Backbone**:
   - Discretizes continuous state-space matrices ($\bar{A}, \bar{B}, C$) dynamically conditioned on text tags.
   - **System 2 Hamiltonian Latent Deliberation Core**: Multi-step ODE shooting along the acoustic manifold to refine transient dynamics and sub-bass resonance.
   - **Multi-Codebook RVQ Head**: 8 parallel projection heads matching EnCodec 24kHz RVQ representations.

2. **Real Dataset Ingestion**:
   - Trained on real studio drum one-shots from Hugging Face (`yojul/one-shot-hip-hop-drums`: 19,673 samples across 808s, kicks, snares, claps, hi-hats, open hats, cymbals).
   - Dynamic acoustic feature extraction (spectral centroid, transient onset, sub-energy ratios, decay envelopes) constructing rich multi-attribute conditioning prompts.

3. **Android NDK Rust Engine**:
   - Zero C++ dependency native Rust engine compiled for `aarch64-linux-android` and `x86_64-linux-android`.
   - $O(1)$ constant-time recurrent scan state cache (`TGSSMStateCache`).
   - Benchmarked on physical **Google Pixel 10a** at **`22.62 tokens/sec`** with **`49.5 ms`** System 2 deliberation.

---

## 🎧 Generated Audio Samples

Located in `generated_audio/`:

* **`test_808_kick.wav`**: Prompt: *"808, sub kick, deep low end, heavy sub, 50Hz sub punch, hard transient"* (`24kHz`, `0.64s`)
* **`test_crisp_snare.wav`**: Prompt: *"snare, drum, acoustic snare, bright, crisp, hard transient, snappy wire rattle"* (`24kHz`, `0.64s`)
* **`test_trap_hat.wav`**: Prompt: *"hihat, closed hat, metallic, crisp, fast decay, short tail, trap sizzle"* (`24kHz`, `0.64s`)

---

## 🚀 Quickstart

### 1. Generate Drum Samples from Text Tags (Python / RTX 3060)

```bash
# Generate 808 kick
PYTHONPATH=python python3 python/generate_drum.py \
  --prompt "808, sub kick, deep low end, heavy sub, 50Hz sub punch" \
  --output generated_audio/my_808.wav

# Generate Crisp Snare
PYTHONPATH=python python3 python/generate_drum.py \
  --prompt "snare, drum, acoustic snare, bright, crisp, snappy wire rattle" \
  --output generated_audio/my_snare.wav
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
