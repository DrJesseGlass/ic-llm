#![cfg(feature = "canbench-rs")]

use canbench_rs::{bench, bench_fn, BenchResult};
use std::cell::RefCell;

use crate::qwen3::Qwen3Model;
use ic_dev_kit_rs::candle::CandleModel;
use ic_dev_kit_rs::text_generation::{AutoregressiveModel, GenerationConfig, TokenizerHandle};

const MODEL_SIZE: usize = 341_455_328; // Qwen3-0.6B-allq4k-f16src.gguf
const TOKENIZER_BYTES: &[u8] = include_bytes!("/home/jesse/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/blobs/aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4");

thread_local! {
    static MODEL: RefCell<Option<Qwen3Model>> = RefCell::new(None);
    static TOKENIZER: RefCell<Option<Box<dyn TokenizerHandle>>> = RefCell::new(None);
}

fn ensure_model_loaded() {
    MODEL.with(|m| {
        if m.borrow().is_some() {
            return;
        }

        let mut model_bytes = vec![0u8; MODEL_SIZE];
        ic_cdk::api::stable::stable_read(0, &mut model_bytes);

        let model = Qwen3Model::load(model_bytes, Some(TOKENIZER_BYTES.to_vec()))
            .expect("Failed to load model");

        let tokenizer = model.get_tokenizer();

        *m.borrow_mut() = Some(model);
        TOKENIZER.with(|t| *t.borrow_mut() = Some(tokenizer));
    });
}

// ═══════════════════════════════════════════════════════════════
//  Model Loading
// ═══════════════════════════════════════════════════════════════

#[bench(raw)]
fn bench_model_load() -> BenchResult {
    bench_fn(|| {
        let mut model_bytes = vec![0u8; MODEL_SIZE];
        ic_cdk::api::stable::stable_read(0, &mut model_bytes);

        let _model = Qwen3Model::load(model_bytes, Some(TOKENIZER_BYTES.to_vec()))
            .expect("Failed to load model");
    })
}

// ═══════════════════════════════════════════════════════════════
//  Tokenization
// ═══════════════════════════════════════════════════════════════

#[bench(raw)]
fn bench_tokenize_short() -> BenchResult {
    ensure_model_loaded();

    bench_fn(|| {
        TOKENIZER.with(|t| {
            let tokenizer = t.borrow();
            let tokenizer = tokenizer.as_ref().unwrap();
            let _ = tokenizer.encode("1 + 1 = ");
        })
    })
}

#[bench(raw)]
fn bench_tokenize_medium() -> BenchResult {
    ensure_model_loaded();
    let prompt = "The quick brown fox jumps over the lazy dog. ".repeat(10);

    bench_fn(|| {
        TOKENIZER.with(|t| {
            let tokenizer = t.borrow();
            let tokenizer = tokenizer.as_ref().unwrap();
            let _ = tokenizer.encode(&prompt);
        })
    })
}

#[bench(raw)]
fn bench_tokenize_long() -> BenchResult {
    ensure_model_loaded();
    let prompt = "The quick brown fox jumps over the lazy dog. ".repeat(100);

    bench_fn(|| {
        TOKENIZER.with(|t| {
            let tokenizer = t.borrow();
            let tokenizer = tokenizer.as_ref().unwrap();
            let _ = tokenizer.encode(&prompt);
        })
    })
}

#[bench(raw)]
fn bench_decode_10_tokens() -> BenchResult {
    ensure_model_loaded();
    let tokens: Vec<u32> = vec![791, 4320, 374, 220, 17, 13, 578, 1314, 315, 220]; // sample tokens

    bench_fn(|| {
        TOKENIZER.with(|t| {
            let tokenizer = t.borrow();
            let tokenizer = tokenizer.as_ref().unwrap();
            let _ = tokenizer.decode(&tokens);
        })
    })
}

// ═══════════════════════════════════════════════════════════════
//  Single Token Generation (measures one forward pass)
// ═══════════════════════════════════════════════════════════════

#[bench(raw)]
fn bench_single_token() -> BenchResult {
    ensure_model_loaded();

    bench_fn(|| {
        MODEL.with(|m| {
            let mut model = m.borrow_mut();
            let model = model.as_mut().unwrap();

            TOKENIZER.with(|t| {
                let tokenizer = t.borrow();
                let tokenizer = tokenizer.as_ref().unwrap();

                let config = GenerationConfig {
                    max_tokens: 1,
                    temperature: 0.7,
                    ..Default::default()
                };

                model.reset();
                let _ = model.init_generation(
                    "1 + 1 = ".to_string(),
                    tokenizer.as_ref(),
                    &config
                );
                // init_generation produces the first token
            })
        })
    })
}

// ═══════════════════════════════════════════════════════════════
//  Prefill Cost (different prompt lengths, 1 output token)
// ═══════════════════════════════════════════════════════════════

#[bench(raw)]
fn bench_prefill_short_4_tokens() -> BenchResult {
    ensure_model_loaded();

    bench_fn(|| {
        MODEL.with(|m| {
            let mut model = m.borrow_mut();
            let model = model.as_mut().unwrap();

            TOKENIZER.with(|t| {
                let tokenizer = t.borrow();
                let tokenizer = tokenizer.as_ref().unwrap();

                let config = GenerationConfig {
                    max_tokens: 1,
                    temperature: 0.7,
                    ..Default::default()
                };

                model.reset();
                let _ = model.init_generation(
                    "Hi!".to_string(), // ~2-4 tokens
                    tokenizer.as_ref(),
                    &config
                );
            })
        })
    })
}

#[bench(raw)]
fn bench_prefill_medium_50_tokens() -> BenchResult {
    ensure_model_loaded();
    let prompt = "The quick brown fox jumps over the lazy dog. ".repeat(5); // ~50 tokens

    bench_fn(|| {
        MODEL.with(|m| {
            let mut model = m.borrow_mut();
            let model = model.as_mut().unwrap();

            TOKENIZER.with(|t| {
                let tokenizer = t.borrow();
                let tokenizer = tokenizer.as_ref().unwrap();

                let config = GenerationConfig {
                    max_tokens: 1,
                    temperature: 0.7,
                    ..Default::default()
                };

                model.reset();
                let _ = model.init_generation(
                    prompt.clone(),
                    tokenizer.as_ref(),
                    &config
                );
            })
        })
    })
}

#[bench(raw)]
fn bench_prefill_long_200_tokens() -> BenchResult {
    ensure_model_loaded();
    let prompt = "The quick brown fox jumps over the lazy dog. ".repeat(20); // ~200 tokens

    bench_fn(|| {
        MODEL.with(|m| {
            let mut model = m.borrow_mut();
            let model = model.as_mut().unwrap();

            TOKENIZER.with(|t| {
                let tokenizer = t.borrow();
                let tokenizer = tokenizer.as_ref().unwrap();

                let config = GenerationConfig {
                    max_tokens: 1,
                    temperature: 0.7,
                    ..Default::default()
                };

                model.reset();
                let _ = model.init_generation(
                    prompt.clone(),
                    tokenizer.as_ref(),
                    &config
                );
            })
        })
    })
}

// ═══════════════════════════════════════════════════════════════
//  Generation Cost (fixed short prompt, varying output tokens)
// ═══════════════════════════════════════════════════════════════

#[bench(raw)]
fn bench_generate_1_token() -> BenchResult {
    ensure_model_loaded();

    bench_fn(|| {
        MODEL.with(|m| {
            let mut model = m.borrow_mut();
            let model = model.as_mut().unwrap();

            TOKENIZER.with(|t| {
                let tokenizer = t.borrow();
                let tokenizer = tokenizer.as_ref().unwrap();

                let config = GenerationConfig {
                    max_tokens: 1,
                    temperature: 0.7,
                    ..Default::default()
                };

                model.reset();
                let _ = model.init_generation(
                    "1 + 1 = ".to_string(),
                    tokenizer.as_ref(),
                    &config
                );
            })
        })
    })
}

#[bench(raw)]
fn bench_generate_3_tokens() -> BenchResult {
    ensure_model_loaded();

    bench_fn(|| {
        MODEL.with(|m| {
            let mut model = m.borrow_mut();
            let model = model.as_mut().unwrap();

            TOKENIZER.with(|t| {
                let tokenizer = t.borrow();
                let tokenizer = tokenizer.as_ref().unwrap();

                let config = GenerationConfig {
                    max_tokens: 3,
                    temperature: 0.7,
                    ..Default::default()
                };

                model.reset();
                let _ = model.init_generation(
                    "1 + 1 = ".to_string(),
                    tokenizer.as_ref(),
                    &config
                );

                for _ in 1..3 {
                    if model.is_generation_complete() { break; }
                    let _ = model.generate_next_token(tokenizer.as_ref());
                }
            })
        })
    })
}

#[bench(raw)]
fn bench_generate_5_tokens() -> BenchResult {
    ensure_model_loaded();

    bench_fn(|| {
        MODEL.with(|m| {
            let mut model = m.borrow_mut();
            let model = model.as_mut().unwrap();

            TOKENIZER.with(|t| {
                let tokenizer = t.borrow();
                let tokenizer = tokenizer.as_ref().unwrap();

                let config = GenerationConfig {
                    max_tokens: 5,
                    temperature: 0.7,
                    ..Default::default()
                };

                model.reset();
                let _ = model.init_generation(
                    "1 + 1 = ".to_string(),
                    tokenizer.as_ref(),
                    &config
                );

                for _ in 1..5 {
                    if model.is_generation_complete() { break; }
                    let _ = model.generate_next_token(tokenizer.as_ref());
                }
            })
        })
    })
}