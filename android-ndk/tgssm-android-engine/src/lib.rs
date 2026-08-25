use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use byteorder::{LittleEndian, ReadBytesExt};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub d_model: usize,
    pub n_layers: usize,
    pub d_state: usize,
    pub d_conv: usize,
    pub expand: usize,
    pub dt_rank: usize,
    pub num_experts: usize,
    pub top_k_experts: usize,
    pub deliberation_steps: usize,
    pub deliberation_horizon: usize,
    pub vocab_size: usize,
}

#[inline(always)]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline(always)]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

pub fn rmsnorm(x: &[f32], weight: &[f32], out: &mut [f32], eps: f32) {
    let d = x.len();
    let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
    let scale = 1.0 / (sum_sq / d as f32 + eps).sqrt();
    for i in 0..d {
        out[i] = x[i] * scale * weight[i];
    }
}

pub fn matmul_vec(x: &[f32], w: &[f32], out: &mut [f32], k: usize, n: usize) {
    assert_eq!(x.len(), k);
    assert_eq!(w.len(), n * k);
    assert_eq!(out.len(), n);

    for j in 0..n {
        let row_offset = j * k;
        let mut sum = 0.0f32;
        let mut i = 0;
        while i + 3 < k {
            sum += x[i] * w[row_offset + i]
                + x[i + 1] * w[row_offset + i + 1]
                + x[i + 2] * w[row_offset + i + 2]
                + x[i + 3] * w[row_offset + i + 3];
            i += 4;
        }
        while i < k {
            sum += x[i] * w[row_offset + i];
            i += 1;
        }
        out[j] = sum;
    }
}

pub fn matmul_vec_bias(x: &[f32], w: &[f32], bias: &[f32], out: &mut [f32], k: usize, n: usize) {
    matmul_vec(x, w, out, k, n);
    for j in 0..n {
        out[j] += bias[j];
    }
}

pub fn softmax(logits: &mut [f32]) {
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in logits.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    let inv_sum = 1.0 / (sum + 1e-8);
    for v in logits.iter_mut() {
        *v *= inv_sum;
    }
}

#[derive(Clone)]
pub struct TensorData {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

pub struct SelectiveSSMLayer {
    pub in_proj_w: Vec<f32>,       // [2 * d_inner, d_model]
    pub conv1d_w: Vec<f32>,        // [d_inner, 1, d_conv]
    pub conv1d_b: Option<Vec<f32>>,// [d_inner]
    pub x_proj_w: Vec<f32>,        // [dt_rank + 2*d_state, d_inner]
    pub dt_proj_w: Vec<f32>,       // [d_inner, dt_rank]
    pub dt_proj_b: Vec<f32>,       // [d_inner]
    pub a_log: Vec<f32>,           // [d_inner, d_state]
    pub d_param: Vec<f32>,         // [d_inner]
    pub out_proj_w: Vec<f32>,      // [d_model, d_inner]
}

pub struct ExpertFFNLayer {
    pub w1: Vec<f32>, // [d_ffn, d_model]
    pub w2: Vec<f32>, // [d_model, d_ffn]
    pub w3: Vec<f32>, // [d_ffn, d_model]
}

pub struct MetabolicMoELayer {
    pub router_w: Vec<f32>, // [num_experts, d_model]
    pub experts: Vec<ExpertFFNLayer>,
}

pub struct TGSSMBlockWeights {
    pub norm1: Vec<f32>, // [d_model]
    pub ssm: SelectiveSSMLayer,
    pub norm2: Vec<f32>, // [d_model]
    pub moe: MetabolicMoELayer,
}

pub struct LatentDeliberationWeights {
    pub metric_norm: Vec<f32>,
    pub metric_w1: Vec<f32>,
    pub metric_b1: Vec<f32>,
    pub metric_w2: Vec<f32>,
    pub prop_norm: Vec<f32>,
    pub prop_w1: Vec<f32>,
    pub prop_b1: Vec<f32>,
    pub prop_w2: Vec<f32>,
    pub dt: f32,
}

pub struct TGSSMModel {
    pub config: ModelConfig,
    pub tok_embeddings: Vec<f32>, // [vocab_size, d_model]
    pub layers: Vec<TGSSMBlockWeights>,
    pub final_norm: Vec<f32>,     // [d_model]
    pub deliberation: LatentDeliberationWeights,
    pub target_proj_norm: Vec<f32>,
    pub target_proj_w: Vec<f32>,  // [d_model, d_model]
    pub lm_head_w: Vec<f32>,      // [vocab_size, d_model]
}

#[derive(Clone)]
pub struct LayerRecurrentState {
    pub conv_state: Vec<f32>, // [d_inner * d_conv]
    pub ssm_state: Vec<f32>,  // [d_inner * d_state]
}

#[derive(Clone)]
pub struct TGSSMStateCache {
    pub layers: Vec<LayerRecurrentState>,
}

impl TGSSMStateCache {
    pub fn new(config: &ModelConfig) -> Self {
        let d_inner = config.d_model * config.expand;
        let mut layers = Vec::with_capacity(config.n_layers);
        for _ in 0..config.n_layers {
            layers.push(LayerRecurrentState {
                conv_state: vec![0.0f32; d_inner * config.d_conv],
                ssm_state: vec![0.0f32; d_inner * config.d_state],
            });
        }
        Self { layers }
    }
}

fn pop_tensor(map: &mut HashMap<String, TensorData>, name: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    map.remove(name)
        .map(|t| t.data)
        .ok_or_else(|| format!("Missing tensor: {}", name).into())
}

impl TGSSMModel {
    pub fn load_from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != b"TGSSM001" {
            return Err("Invalid TGSSM binary magic header".into());
        }

        let cfg_len = reader.read_u32::<LittleEndian>()? as usize;
        let mut cfg_buf = vec![0u8; cfg_len];
        reader.read_exact(&mut cfg_buf)?;
        let config: ModelConfig = serde_json::from_slice(&cfg_buf)?;

        let num_tensors = reader.read_u32::<LittleEndian>()? as usize;
        let mut tensors = HashMap::with_capacity(num_tensors);

        for _ in 0..num_tensors {
            let name_len = reader.read_u16::<LittleEndian>()? as usize;
            let mut name_buf = vec![0u8; name_len];
            reader.read_exact(&mut name_buf)?;
            let name = String::from_utf8(name_buf)?;

            let ndims = reader.read_u16::<LittleEndian>()? as usize;
            let mut shape = Vec::with_capacity(ndims);
            for _ in 0..ndims {
                shape.push(reader.read_u32::<LittleEndian>()? as usize);
            }

            let numel = reader.read_u32::<LittleEndian>()? as usize;
            let mut data = vec![0.0f32; numel];
            reader.read_f32_into::<LittleEndian>(&mut data)?;

            tensors.insert(name, TensorData { shape, data });
        }

        let tok_embeddings = pop_tensor(&mut tensors, "tok_embeddings.weight")?;
        let final_norm = pop_tensor(&mut tensors, "final_norm.weight")?;
        let target_proj_norm = pop_tensor(&mut tensors, "target_projector.0.weight")?;
        let target_proj_w = pop_tensor(&mut tensors, "target_projector.1.weight")?;
        let lm_head_w = pop_tensor(&mut tensors, "lm_head.weight")?;

        let deliberation = LatentDeliberationWeights {
            metric_norm: pop_tensor(&mut tensors, "deliberation_core.metric_field.0.weight")?,
            metric_w1: pop_tensor(&mut tensors, "deliberation_core.metric_field.1.weight")?,
            metric_b1: pop_tensor(&mut tensors, "deliberation_core.metric_field.1.bias")?,
            metric_w2: pop_tensor(&mut tensors, "deliberation_core.metric_field.3.weight")?,
            prop_norm: pop_tensor(&mut tensors, "deliberation_core.propagator.0.weight")?,
            prop_w1: pop_tensor(&mut tensors, "deliberation_core.propagator.1.weight")?,
            prop_b1: pop_tensor(&mut tensors, "deliberation_core.propagator.1.bias")?,
            prop_w2: pop_tensor(&mut tensors, "deliberation_core.propagator.3.weight")?,
            dt: pop_tensor(&mut tensors, "deliberation_core.dt")?[0],
        };

        let mut layers = Vec::with_capacity(config.n_layers);
        for l in 0..config.n_layers {
            let prefix = format!("layers.{}", l);
            let norm1 = pop_tensor(&mut tensors, &format!("{}.norm1.weight", prefix))?;
            let norm2 = pop_tensor(&mut tensors, &format!("{}.norm2.weight", prefix))?;

            let ssm = SelectiveSSMLayer {
                in_proj_w: pop_tensor(&mut tensors, &format!("{}.ssm.in_proj.weight", prefix))?,
                conv1d_w: pop_tensor(&mut tensors, &format!("{}.ssm.conv1d.weight", prefix))?,
                conv1d_b: tensors.remove(&format!("{}.ssm.conv1d.bias", prefix)).map(|t| t.data),
                x_proj_w: pop_tensor(&mut tensors, &format!("{}.ssm.x_proj.weight", prefix))?,
                dt_proj_w: pop_tensor(&mut tensors, &format!("{}.ssm.dt_proj.weight", prefix))?,
                dt_proj_b: pop_tensor(&mut tensors, &format!("{}.ssm.dt_proj.bias", prefix))?,
                a_log: pop_tensor(&mut tensors, &format!("{}.ssm.A_log", prefix))?,
                d_param: pop_tensor(&mut tensors, &format!("{}.ssm.D", prefix))?,
                out_proj_w: pop_tensor(&mut tensors, &format!("{}.ssm.out_proj.weight", prefix))?,
            };

            let router_w = pop_tensor(&mut tensors, &format!("{}.moe.router.weight", prefix))?;
            let mut experts = Vec::with_capacity(config.num_experts);
            for e in 0..config.num_experts {
                experts.push(ExpertFFNLayer {
                    w1: pop_tensor(&mut tensors, &format!("{}.moe.experts.{}.w1.weight", prefix, e))?,
                    w2: pop_tensor(&mut tensors, &format!("{}.moe.experts.{}.w2.weight", prefix, e))?,
                    w3: pop_tensor(&mut tensors, &format!("{}.moe.experts.{}.w3.weight", prefix, e))?,
                });
            }

            let moe = MetabolicMoELayer { router_w, experts };
            layers.push(TGSSMBlockWeights { norm1, ssm, norm2, moe });
        }

        Ok(Self {
            config,
            tok_embeddings,
            layers,
            final_norm,
            deliberation,
            target_proj_norm,
            target_proj_w,
            lm_head_w,
        })
    }

    /// Single O(1) step forward with recurrent state cache
    pub fn step_single_token(
        &self,
        token_id: u32,
        cache: &mut TGSSMStateCache,
    ) -> (Vec<f32>, Vec<f32>) {
        let d_model = self.config.d_model;
        let d_inner = d_model * self.config.expand;
        let d_state = self.config.d_state;
        let dt_rank = self.config.dt_rank;
        let d_conv = self.config.d_conv;
        let d_ffn = (d_model as f32 * 2.5) as usize;

        let tok_idx = (token_id as usize).min(self.config.vocab_size - 1);
        let offset = tok_idx * d_model;
        let mut x = self.tok_embeddings[offset..offset + d_model].to_vec();

        let mut xz_buf = vec![0.0f32; 2 * d_inner];
        let mut x_proj_out = vec![0.0f32; dt_rank + 2 * d_state];
        let mut dt_buf = vec![0.0f32; d_inner];
        let mut y_gated = vec![0.0f32; d_inner];
        let mut ssm_out = vec![0.0f32; d_model];
        let mut norm_buf = vec![0.0f32; d_model];

        for l in 0..self.config.n_layers {
            let block = &self.layers[l];
            let layer_state = &mut cache.layers[l];

            // --- 1. SSM Branch ---
            rmsnorm(&x, &block.norm1, &mut norm_buf, 1e-5);
            matmul_vec(&norm_buf, &block.ssm.in_proj_w, &mut xz_buf, d_model, 2 * d_inner);

            let x_val = &xz_buf[0..d_inner];
            let z_val = &xz_buf[d_inner..2 * d_inner];

            // 1D Causal Conv1d update
            let mut x_act = vec![0.0f32; d_inner];
            for c in 0..d_inner {
                let conv_offset = c * d_conv;
                // Shift buffer left
                for k in 0..d_conv - 1 {
                    layer_state.conv_state[conv_offset + k] = layer_state.conv_state[conv_offset + k + 1];
                }
                layer_state.conv_state[conv_offset + d_conv - 1] = x_val[c];

                let mut conv_sum = 0.0f32;
                if let Some(ref bias) = block.ssm.conv1d_b {
                    conv_sum = bias[c];
                }
                for k in 0..d_conv {
                    conv_sum += layer_state.conv_state[conv_offset + k] * block.ssm.conv1d_w[conv_offset + k];
                }
                x_act[c] = silu(conv_sum);
            }

            // Parameter projection
            matmul_vec(&x_act, &block.ssm.x_proj_w, &mut x_proj_out, d_inner, dt_rank + 2 * d_state);
            let dt_raw = &x_proj_out[0..dt_rank];
            let b_raw = &x_proj_out[dt_rank..dt_rank + d_state];
            let c_raw = &x_proj_out[dt_rank + d_state..dt_rank + 2 * d_state];

            matmul_vec_bias(dt_raw, &block.ssm.dt_proj_w, &block.ssm.dt_proj_b, &mut dt_buf, dt_rank, d_inner);

            // SSM Recurrent update
            for i in 0..d_inner {
                let dt_v = softplus(dt_buf[i]);
                let x_v = x_act[i];
                let mut sum_c_h = 0.0f32;

                for n in 0..d_state {
                    let a_log_v = block.ssm.a_log[i * d_state + n];
                    let a_v = -a_log_v.exp();
                    let da = (dt_v * a_v).exp();
                    let db = dt_v * b_raw[n];

                    let h_idx = i * d_state + n;
                    layer_state.ssm_state[h_idx] = da * layer_state.ssm_state[h_idx] + db * x_v;
                    sum_c_h += c_raw[n] * layer_state.ssm_state[h_idx];
                }

                let y_val = sum_c_h + block.ssm.d_param[i] * x_v;
                y_gated[i] = y_val * silu(z_val[i]);
            }

            matmul_vec(&y_gated, &block.ssm.out_proj_w, &mut ssm_out, d_inner, d_model);
            for i in 0..d_model {
                x[i] += ssm_out[i];
            }

            // --- 2. MoE Branch ---
            rmsnorm(&x, &block.norm2, &mut norm_buf, 1e-5);
            let mut router_logits = vec![0.0f32; self.config.num_experts];
            matmul_vec(&norm_buf, &block.moe.router_w, &mut router_logits, d_model, self.config.num_experts);
            softmax(&mut router_logits);

            let mut indexed_probs: Vec<(usize, f32)> = router_logits.iter().cloned().enumerate().collect();
            indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let top_k = self.config.top_k_experts.min(self.config.num_experts);
            let top_k_items = &indexed_probs[0..top_k];
            let sum_prob: f32 = top_k_items.iter().map(|(_, p)| *p).sum();
            let norm_factor = 1.0 / (sum_prob + 1e-8);

            let mut w1_out = vec![0.0f32; d_ffn];
            let mut w3_out = vec![0.0f32; d_ffn];
            let mut swiglu_act = vec![0.0f32; d_ffn];
            let mut expert_out = vec![0.0f32; d_model];

            for &(expert_idx, raw_prob) in top_k_items {
                let weight = raw_prob * norm_factor;
                let expert = &block.moe.experts[expert_idx];

                matmul_vec(&norm_buf, &expert.w1, &mut w1_out, d_model, d_ffn);
                matmul_vec(&norm_buf, &expert.w3, &mut w3_out, d_model, d_ffn);

                for i in 0..d_ffn {
                    swiglu_act[i] = silu(w1_out[i]) * w3_out[i];
                }

                matmul_vec(&swiglu_act, &expert.w2, &mut expert_out, d_ffn, d_model);

                for i in 0..d_model {
                    x[i] += weight * expert_out[i];
                }
            }
        }

        // Final Norm
        let mut final_z = vec![0.0f32; d_model];
        rmsnorm(&x, &self.final_norm, &mut final_z, 1e-5);

        // LM Head Logits
        let mut logits = vec![0.0f32; self.config.vocab_size];
        matmul_vec(&final_z, &self.lm_head_w, &mut logits, d_model, self.config.vocab_size);

        (final_z, logits)
    }

    pub fn forward_deliberation(&self, z_t: &[f32], horizon: usize) -> Vec<Vec<f32>> {
        let d_model = self.config.d_model;
        let delib = &self.deliberation;
        let dt = softplus(delib.dt).clamp(0.01, 0.5);

        let mut norm_z = vec![0.0f32; d_model];
        rmsnorm(z_t, &delib.metric_norm, &mut norm_z, 1e-5);

        let mut h1 = vec![0.0f32; d_model];
        matmul_vec_bias(&norm_z, &delib.metric_w1, &delib.metric_b1, &mut h1, d_model, d_model);
        for v in h1.iter_mut() {
            *v = silu(*v);
        }

        let mut v = vec![0.0f32; d_model];
        matmul_vec(&h1, &delib.metric_w2, &mut v, d_model, d_model);

        let mut z = z_t.to_vec();
        let mut rollout = Vec::with_capacity(horizon);

        let mut zv = vec![0.0f32; 2 * d_model];
        let mut norm_zv = vec![0.0f32; 2 * d_model];
        let mut prop_h1 = vec![0.0f32; 2 * d_model];
        let mut force = vec![0.0f32; d_model];

        for _ in 0..horizon {
            for _ in 0..self.config.deliberation_steps {
                zv[0..d_model].copy_from_slice(&z);
                zv[d_model..2 * d_model].copy_from_slice(&v);

                rmsnorm(&zv, &delib.prop_norm, &mut norm_zv, 1e-5);
                matmul_vec_bias(&norm_zv, &delib.prop_w1, &delib.prop_b1, &mut prop_h1, 2 * d_model, 2 * d_model);
                for val in prop_h1.iter_mut() {
                    *val = silu(*val);
                }
                matmul_vec(&prop_h1, &delib.prop_w2, &mut force, 2 * d_model, d_model);

                for i in 0..d_model {
                    v[i] += dt * force[i];
                }
            }

            for i in 0..d_model {
                z[i] += dt * v[i];
            }
            rollout.push(z.clone());
        }

        rollout
    }

    pub fn sample_next_token(logits: &[f32], temperature: f32, top_k: usize) -> u32 {
        let mut temp_logits = logits.to_vec();
        let temp = temperature.max(1e-4);
        for l in temp_logits.iter_mut() {
            *l /= temp;
        }

        if top_k > 0 && top_k < temp_logits.len() {
            let mut sorted: Vec<(usize, f32)> = temp_logits.iter().cloned().enumerate().collect();
            sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let threshold = sorted[top_k - 1].1;
            for l in temp_logits.iter_mut() {
                if *l < threshold {
                    *l = f32::NEG_INFINITY;
                }
            }
        }

        softmax(&mut temp_logits);

        let r: f32 = rand::random();
        let mut cdf = 0.0f32;
        for (idx, &prob) in temp_logits.iter().enumerate() {
            cdf += prob;
            if r <= cdf {
                return idx as u32;
            }
        }

        (temp_logits.len() - 1) as u32
    }
}
