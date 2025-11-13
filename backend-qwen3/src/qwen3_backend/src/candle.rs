use std::cell::RefCell;
use serde::Deserialize;
use candid::CandidType;
use candle_core::{DType, Device, Tensor};
use candle_core::quantized::gguf_file;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_qwen3::{ModelWeights as QuantizedQwen3};
use tokenizers::Tokenizer;
use std::io::Cursor;
use anyhow::{anyhow, Result};

thread_local! {
    static QWEN_MODEL: RefCell<Option<QwenModel>> = RefCell::new(None);
}

struct QwenModel {
    model: QuantizedQwen3,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    tokens: Vec<u32>,
    repeat_penalty: f32,
    repeat_last_n: usize,
    eos_token: u32,
}

#[derive(CandidType, Deserialize)]
pub enum EmptyResult {
    Ok,
    Err(String),
}

fn internal_setup_model() -> Result<(), anyhow::Error> {
    let device = Device::Cpu;

    ic_cdk::println!("Loading Qwen3 model from stable storage...");

    // Get the GGUF weights from stable storage
    let weights = match crate::load_bytes_from_stable("model_weights") {
        Some(bytes) => bytes,
        None => return Err(anyhow!("Tokenizer not found in stable storage")),
    };

    ic_cdk::println!("Weights size: {} bytes ({:.2} MB)",
        weights.len(),
        weights.len() as f64 / 1_048_576.0
    );

    // Get tokenizer from stable storage
    let tokenizer_bytes = match crate::load_bytes_from_stable("tokenizer") {
        Some(bytes) => bytes,
        None => return Err(anyhow!("Tokenizer not found in stable storage")),
    };

    let tokenizer = Tokenizer::from_bytes(&tokenizer_bytes)
        .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

    // Get EOS token
    let eos_token = match tokenizer.get_vocab(true).get("<|endoftext|>") {
        Some(&token) => token,
        None => match tokenizer.get_vocab(true).get("<|im_end|>") {
            Some(&token) => token,
            None => {
                ic_cdk::println!("Warning: no EOS token found, using 0");
                0
            }
        }
    };

    // Load GGUF quantized model
    let mut cursor = Cursor::new(weights);
    let content = gguf_file::Content::read(&mut cursor)
        .map_err(|e| anyhow!("Failed to read GGUF: {}", e))?;

    ic_cdk::println!("GGUF file parsed, loading model weights...");

    #[cfg(feature = "simd-flash-attn")]
    ic_cdk::println!("Loading with SIMD flash attention");

    let model = QuantizedQwen3::from_gguf(content, &mut cursor, &device)?;

    ic_cdk::println!("✅ Qwen3 model loaded successfully");

    let logits_processor = LogitsProcessor::new(299792458, None, None);

    let qwen_model = QwenModel {
        model,
        tokenizer,
        tokens: vec![],
        logits_processor,
        repeat_penalty: 1.,
        repeat_last_n: 64,
        eos_token,
    };

    QWEN_MODEL.with(|cell| {
        *cell.borrow_mut() = Some(qwen_model);
    });

    Ok(())
}

#[ic_cdk::update]
pub fn setup_model() -> EmptyResult {
    match internal_setup_model() {
        Ok(_) => EmptyResult::Ok,
        Err(e) => EmptyResult::Err(e.to_string()),
    }
}

// Generation config
#[derive(CandidType, Deserialize, Clone)]
pub struct GenerationConfig {
    pub temperature: f64,
    pub top_p: f64,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub seed: u64,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            seed: 42,
        }
    }
}

#[derive(CandidType, Deserialize)]
pub struct InferenceRequest {
    pub prompt: String,
    pub config: Option<GenerationConfig>,
}

#[derive(CandidType, Deserialize)]
pub struct InferenceResponse {
    pub generated_text: String,
    pub tokens_generated: usize,
    pub instructions_used: u64,
    pub success: bool,
    pub error: Option<String>,
}

// Internal inference function
fn internal_generate(request: InferenceRequest) -> Result<InferenceResponse, anyhow::Error> {
    let start_instructions = ic_cdk::api::performance_counter(0);
    let config = request.config.unwrap_or_default();

    QWEN_MODEL.with(|cell| -> Result<InferenceResponse, anyhow::Error> {
        let mut model_ref = cell.borrow_mut();
        let model = model_ref.as_mut()
            .ok_or_else(|| anyhow!("Model not initialized. Call setup_model first."))?;

        // Reset state
        //model.clear_kv_cache();
        model.tokens.clear();

        // Setup generation parameters
        let temp = if config.temperature <= 0. { None } else { Some(config.temperature) };
        let top_p = if config.top_p <= 0. || config.top_p >= 1. { None } else { Some(config.top_p) };

        model.logits_processor = LogitsProcessor::new(config.seed, temp, top_p);
        model.repeat_penalty = config.repeat_penalty;
        model.repeat_last_n = config.repeat_last_n;

        // Tokenize prompt
        let tokens = model.tokenizer
            .encode(request.prompt.clone(), true)
            .map_err(|e| anyhow!("Tokenization error: {}", e))?
            .get_ids()
            .to_vec();

        ic_cdk::println!("Prompt encoded to {} tokens", tokens.len());

        // Process prompt and generate first token
        let first_token = process_tokens(model, &tokens)?;
        let mut generated_text = first_token;

        // Generate additional tokens (up to 50 for now to avoid instruction limits)
        let max_tokens = 50;
        for _ in 0..max_tokens {
            if model.tokens.last().map_or(false, |&t| t == model.eos_token) {
                break;
            }

            // Check instruction limit
            let instructions_so_far = ic_cdk::api::performance_counter(0) - start_instructions;
            if instructions_so_far > 30_000_000_000 {
                ic_cdk::println!("Approaching instruction limit, stopping generation");
                break;
            }

            let last_token = *model.tokens.last().unwrap();
            let token_text = process_tokens(model, &[last_token])?;
            generated_text.push_str(&token_text);
        }

        let instructions_used = ic_cdk::api::performance_counter(0) - start_instructions;

        Ok(InferenceResponse {
            generated_text,
            tokens_generated: model.tokens.len(),
            instructions_used,
            success: true,
            error: None,
        })
    })
}

fn process_tokens(model: &mut QwenModel, tokens: &[u32]) -> Result<String, anyhow::Error> {
    let device = Device::Cpu;
    let input = Tensor::new(tokens, &device)
        .map_err(|e| anyhow!("Tensor creation error: {}", e))?
        .unsqueeze(0)
        .map_err(|e| anyhow!("Unsqueeze error: {}", e))?;

    let offset = model.tokens.len();

    let logits = model.model.forward(&input, offset)
        .map_err(|e| anyhow!("Forward pass error: {}", e))?;

    let logits = logits.squeeze(0)
        .map_err(|e| anyhow!("Squeeze error: {}", e))?
        .to_dtype(DType::F32)
        .map_err(|e| anyhow!("DType conversion error: {}", e))?;

    // Apply repeat penalty
    let logits = if model.repeat_penalty == 1. {
        logits
    } else {
        let start_at = model.tokens.len().saturating_sub(model.repeat_last_n);
        let context = &model.tokens[start_at..];
        candle_transformers::utils::apply_repeat_penalty(&logits, model.repeat_penalty, context)
            .map_err(|e| anyhow!("Repeat penalty error: {}", e))?
    };

    let next_token = model.logits_processor.sample(&logits)
        .map_err(|e| anyhow!("Sampling error: {}", e))?;

    model.tokens.push(next_token);

    let token_text = model.tokenizer.decode(&[next_token], false)
        .map_err(|e| anyhow!("Decode error: {}", e))?;

    Ok(token_text)
}

// Public API endpoint
#[ic_cdk::update]
pub fn generate(request: InferenceRequest) -> InferenceResponse {
    match internal_generate(request) {
        Ok(response) => response,
        Err(e) => InferenceResponse {
            generated_text: String::new(),
            tokens_generated: 0,
            instructions_used: 0,
            success: false,
            error: Some(e.to_string()),
        },
    }
}

// Reset generation state
#[ic_cdk::update]
pub fn reset_generation() -> EmptyResult {
    QWEN_MODEL.with(|cell| {
        if let Some(model) = cell.borrow_mut().as_mut() {
            model.tokens.clear();
            //model.clear_kv_cache();
            EmptyResult::Ok
        } else {
            EmptyResult::Err("Model not initialized".to_string())
        }
    })
}

// Check if model is loaded
#[ic_cdk::query]
pub fn is_model_loaded() -> bool {
    QWEN_MODEL.with(|cell| cell.borrow().is_some())
}

// Get model info
#[derive(CandidType, Deserialize)]
pub struct ModelInfo {
    pub loaded: bool,
    pub current_tokens: usize,
}

#[ic_cdk::query]
pub fn get_model_info() -> ModelInfo {
    QWEN_MODEL.with(|cell| {
        if let Some(model) = cell.borrow().as_ref() {
            ModelInfo {
                loaded: true,
                current_tokens: model.tokens.len(),
            }
        } else {
            ModelInfo {
                loaded: false,
                current_tokens: 0,
            }
        }
    })
}