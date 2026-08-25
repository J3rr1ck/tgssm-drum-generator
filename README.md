# 🥁 TG-SSM: Continuous Flow-Matching Multi-Genre Drum Waveform Synthesis Engine

Continuous State-Space Model (**TG-SSM**) adapted for text-conditioned **44.1kHz studio-grade multi-genre drum sample waveform synthesis** via **Optimal Transport Flow Matching (Rectified Flow)** and **DAC (Descript Audio Codec)**.

Supported and trained on local **NVIDIA GeForce RTX 3060** & **Google AI Ultra / Colab Pro (A100, H100, L4)**.

---

## 🌟 Multi-Genre Architecture & Highlights

1. **Continuous Optimal Transport Flow Matching (Rectified Flow)**:
   - Operates directly in the continuous 1024-dimensional acoustic manifold of **DAC 44.1kHz**, completely eliminating discrete codebook quantization distortion, robotic phase jitter, and smeared transients.
   - Vector field velocity formulation: $\mathcal{L}_{\text{FM}} = \| v_\theta(z_t, t, c) - (z_1 - z_0) \|^2$.
   - **System 2 Hamiltonian Latent Deliberation Core**: Multi-step continuous ODE propagation smoothing transient attacks and resonant sub-bass drops.

2. **Diverse Multi-Genre Dataset Mixer**:
   - Ingests and blends multiple real studio datasets:
     * **Trap, Drill, Boom Bap & 808s**: `yojul/one-shot-hip-hop-drums` (19,673 samples)
     * **Acoustic Rock, Indie Jazz & Funk Kits**: `airasoul/drum-kit` (2,700 live samples)
     * **Synthwave, 80s Retro, Techno & House**: Analog physical models (TR-808, TR-909, LinnDrum, CR-78)
     * **Latin Percussion & Afrobeat**: Cowbells, congas, claves, woodblocks, shakers, handclaps
   - Dynamic acoustic feature extraction (spectral centroid, transient onset, sub-energy ratios, decay envelopes) constructing rich multi-attribute conditioning prompts.

3. **Google AI Ultra / Colab Pro Ready**:
   - Includes full Jupyter Notebook `notebooks/TG_SSM_MultiGenre_Drum_Studio_Training.ipynb` with interactive UI sliders and real-time audio playback widgets.

---

## 🎧 Master Audio Previews (44.1kHz Studio Quality)

Located in `generated_audio/`:

| Master Audio Sample | Target Drum / Conditioning Prompt | Sample Rate | Duration | RMS Energy |
| :--- | :--- | :--- | :--- | :--- |
| **`dac_flow_epoch_35_sample_1.wav`** | *"808, sub kick, sub bass, deep sub, deep 50Hz sub, heavy low end, hard transient click"* | `44,100 Hz` | `0.50s` | **`0.2998`** |
| **`dac_flow_epoch_35_sample_2.wav`** | *"snare drum, acoustic snare, crack, bright top end, crisp sheen, hard transient click"* | `44,100 Hz` | `0.50s` | `0.1194` |
| **`dac_flow_epoch_35_sample_3.wav`** | *"hihat, closed hat, metallic, bright top end, crisp sheen, fast decay, short tail"* | `44,100 Hz` | `0.50s` | `0.2033` |
| **`dac_flow_epoch_35_sample_4.wav`** | *"handclap, clap, percussion, warm analog body, punchy mid"* | `44,100 Hz` | `0.50s` | `0.1415` |

---

## ⚡ Google AI Ultra / Colab Training

Open the notebook in Colab with an A100 or H100 GPU:
[`notebooks/TG_SSM_MultiGenre_Drum_Studio_Training.ipynb`](notebooks/TG_SSM_MultiGenre_Drum_Studio_Training.ipynb)

Features:
- Instant dependency setup & DAC 44.1kHz pretrained vocoder loading
- Auto-scaled multi-genre dataset ingestion
- Interactive UI widget with prompt inputs and direct browser audio playback

---

## 🚀 Local CLI Synthesis

Generate any multi-genre drum waveform from the terminal:

```bash
# Trap 808 Sub Kick
PYTHONPATH=python python3 python/generate_dac_drum.py \
  --prompt "trap, 808, sub kick, deep sub, deep 50Hz sub, heavy low end, hard transient click" \
  --output generated_audio/studio_808.wav --steps 30 --cfg 3.0

# Acoustic Rock Snare
PYTHONPATH=python python3 python/generate_dac_drum.py \
  --prompt "acoustic rock, snare drum, live studio crack, bright top end, crisp sheen" \
  --output generated_audio/studio_snare.wav --steps 30 --cfg 3.0

# 80s Synthwave Gated Snare
PYTHONPATH=python python3 python/generate_dac_drum.py \
  --prompt "synthwave, 80s retro, gated snare, analog saturation, diffused room reverb" \
  --output generated_audio/synthwave_snare.wav --steps 30 --cfg 3.0

# Afrobeat / Latin Percussion
PYTHONPATH=python python3 python/generate_dac_drum.py \
  --prompt "latin percussion, afrobeat, cowbell, woodblock, bright top end, metallic" \
  --output generated_audio/afrobeat_cowbell.wav --steps 30 --cfg 3.0
```
