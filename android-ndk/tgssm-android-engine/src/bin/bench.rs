use std::collections::HashMap;
use std::env;
use std::fs;
use std::time::Instant;
use tgssm_android::{TGSSMModel, TGSSMStateCache};

fn load_vocab(path: &str) -> HashMap<u32, String> {
    if let Ok(data) = fs::read_to_string(path) {
        if let Ok(raw_map) = serde_json::from_str::<HashMap<String, String>>(&data) {
            let mut map = HashMap::new();
            for (k, v) in raw_map {
                if let Ok(id) = k.parse::<u32>() {
                    map.insert(id, v);
                }
            }
            return map;
        }
    }
    HashMap::new()
}

fn decode_tokens(tokens: &[u32], vocab: &HashMap<u32, String>) -> String {
    let mut out = String::new();
    for &tok in tokens {
        if let Some(s) = vocab.get(&tok) {
            let clean = s.replace('Ġ', " ").replace('Ċ', "\n");
            out.push_str(&clean);
        } else {
            out.push_str(&format!("<|{}|>", tok));
        }
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!("🚀 TGSSM Foundation Model — Text Generation on Pixel 10a");
    println!("============================================================");

    let args: Vec<String> = env::args().collect();
    let model_path = if args.len() > 1 {
        &args[1]
    } else {
        "/data/local/tmp/tgssm_mobile.bin"
    };
    let vocab_path = if args.len() > 2 {
        &args[2]
    } else {
        "/data/local/tmp/gpt2_vocab.json"
    };

    println!("📂 Loading Model Binary: {}", model_path);
    let start_load = Instant::now();
    let model = TGSSMModel::load_from_file(model_path)?;
    let load_time_ms = start_load.elapsed().as_secs_f32() * 1000.0;
    println!("✅ Model Loaded in {:.2} ms!", load_time_ms);

    println!("📖 Loading Vocabulary Table from: {}", vocab_path);
    let vocab = load_vocab(vocab_path);
    println!("✅ Loaded {} vocabulary tokens!\n", vocab.len());

    let test_prompts = vec![
        (
            "Prompt 1 (Lily's Adventure)",
            vec![7454, 2402, 257, 640, 11, 612, 373, 257, 1310, 2576, 3706, 20037, 508, 6151, 284],
            36,
        ),
        (
            "Prompt 2 (Tim & Spot in the Forest)",
            vec![14967, 290, 465, 3290, 15899, 1816, 656, 262, 1263, 4077, 8222, 284, 1064],
            36,
        ),
        (
            "Prompt 3 (Pip the Bird & Shiny Key)",
            vec![3198, 1110, 11, 257, 1310, 6512, 3706, 25149, 1043, 257, 22441, 1994, 287, 262],
            36,
        ),
    ];

    for (name, prompt_tokens, max_gen) in test_prompts {
        println!("============================================================");
        println!("📝 {}", name);
        println!("============================================================");
        let prompt_text = decode_tokens(&prompt_tokens, &vocab);
        println!("📥 Prompt Text: \"{}\"", prompt_text.trim());

        let mut cache = TGSSMStateCache::new(&model.config);
        let prefill_start = Instant::now();
        let mut last_z = vec![0.0f32; model.config.d_model];
        let mut last_logits = vec![0.0f32; model.config.vocab_size];

        for &tok in &prompt_tokens {
            let (z, logits) = model.step_single_token(tok, &mut cache);
            last_z = z;
            last_logits = logits;
        }
        let first_tok = TGSSMModel::sample_next_token(&last_logits, 0.7, 50);
        let prefill_ms = prefill_start.elapsed().as_secs_f32() * 1000.0;

        // System 2 Deliberation Core
        let delib_start = Instant::now();
        let _rollout = model.forward_deliberation(&last_z, model.config.deliberation_horizon);
        let delib_ms = delib_start.elapsed().as_secs_f32() * 1000.0;

        let mut generated_tokens = Vec::with_capacity(max_gen);
        generated_tokens.push(first_tok);
        let mut curr_tok = first_tok;

        let gen_start = Instant::now();
        for _ in 1..max_gen {
            let (_, logits) = model.step_single_token(curr_tok, &mut cache);
            curr_tok = TGSSMModel::sample_next_token(&logits, 0.7, 50);
            generated_tokens.push(curr_tok);
        }
        let gen_s = gen_start.elapsed().as_secs_f32();
        let tps = (max_gen - 1) as f32 / gen_s;

        let full_text = decode_tokens(&generated_tokens, &vocab);
        println!("📤 Generated Output:");
        println!("\"{}{}\"", prompt_text, full_text);
        println!("------------------------------------------------------------");
        println!("⏱️  Pre-fill: {:.1} ms | 🌀 System 2: {:.1} ms | 🚀 Speed: {:.2} tok/s\n", prefill_ms, delib_ms, tps);
    }

    Ok(())
}
