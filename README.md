# 🥁 TG-SSM: Continuous Flow-Matching Multi-Genre Drum Waveform Synthesis Engine

Continuous State-Space Model (**TG-SSM**) adapted for text-conditioned **44.1kHz studio-grade multi-genre drum sample waveform synthesis** via **Optimal Transport Flow Matching (Rectified Flow)** and **DAC (Descript Audio Codec)**.

---

## 🔬 Expanded 13-Sampler Suite (Stable Audio & Flux.1 Borrowed)

We have expanded the continuous flow integration engine from 4 solvers to **13 production-grade ODE and SDE samplers**:

| Sampler Identifier | Math Order | Source / Provenance | Description & Acoustic Character |
| :--- | :--- | :--- | :--- |
| **`heun`** | **2nd-Order** | Karras / EDM | **Best All-Rounder**: 2nd-order predictor-corrector averaging velocity curvature. |
| **`dpmpp_2m`** | **2nd-Order** | Stable Audio Open | **DPM-Solver++ 2M**: Multistep Adams-Bashforth style interpolation. |
| **`dpmpp_2s`** | **2nd-Order** | Stable Audio Open | **DPM-Solver++ 2S**: Single-step midpoint evaluation. |
| **`flux_rf`** | **1st/2nd-Order** | **Black Forest Labs Flux.1** | **Flux Rectified Flow**: Cosine-shifted schedule warping for sharp bifurcation. |
| **`stable_audio_euler`** | **1st-Order** | **Stability AI (Zach Evans)** | **Exponentially-Warped Flow**: Dynamic density allocation for transient attack. |
| **`euler_ancestral`** | **1st-Order SDE** | Stable Diffusion / Audio | **Euler-Maruyama SDE**: Stochastic Langevin noise churn for organic room texture. |
| **`heun_ancestral`** | **2nd-Order SDE** | Karras / EDM | **Heun SDE**: 2nd-order predictor-corrector with Brownian texture injection. |
| **`dopri5`** | **5th-Order** | Dormand-Prince / Flux.1 | **Highest Precision (RK45)**: 6-stage 5th-order numerical integration. |
| **`bogacki_shampine`** | **3rd-Order** | BS23 / RK23 | **3rd-Order Runge-Kutta**: Balanced harmonic fidelity. |
| **`lms`** | **3rd-Order** | Adams-Bashforth | **Linear Multi-Step**: 3-step velocity history extrapolation. |
| **`rk4`** | **4th-Order** | Classic Runge-Kutta | **Classic 4th-Order**: 4-stage vector integration. |
| **`midpoint`** | **2nd-Order** | RK2 | **Midpoint Runge-Kutta**: Smooth sustained harmonics. |
| **`euler`** | **1st-Order** | Standard Flow | **Linear Step**: Fast draft generation (~5.0s on RTX 3060). |

---

## 🚀 CLI Generation with 13 Samplers

```bash
# 1. Generate with Flux.1 Rectified Flow Cosine-Shifted Schedule
PYTHONPATH=python python3 python/generate_dac_drum.py \
  --prompt "trap, 808, sub kick, deep sub, deep 50Hz sub, heavy low end, hard transient click" \
  --output generated_audio/flux/my_808.wav \
  --sampler flux_rf --steps 30 --cfg 3.0

# 2. Generate with Stable Audio DPM-Solver++ 2M
PYTHONPATH=python python3 python/generate_dac_drum.py \
  --prompt "acoustic rock, snare drum, live studio crack, bright top end, crisp sheen" \
  --output generated_audio/dpmpp/my_snare.wav \
  --sampler dpmpp_2m --steps 25 --cfg 3.0

# 3. Generate with Heun-Ancestral Stochastic SDE (Langevin Texture Churn)
PYTHONPATH=python python3 python/generate_dac_drum.py \
  --prompt "latin percussion, 808 cowbell, metallic cowbell, dual square wave, sharp transient click" \
  --output generated_audio/heun_sde/my_cowbell.wav \
  --sampler heun_ancestral --eta 0.2 --steps 30 --cfg 3.0
```
