use candle_core::{DType, Device, Tensor};
use candle_core::quantized::gguf_file;
use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;
use std::io::Cursor;
use candle_transformers::models::quantized_qwen3::{ModelWeights as QuantizedQwen3, ComputeMode};

/// IC equivalent of console.log
macro_rules! ic_log {
    ($($t:tt)*) => {
        ic_cdk::println!("{}", format_args!($($t)*))
    }
}

pub struct Model {
    model: QuantizedQwen3,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    tokens: Vec<u32>,
    repeat_penalty: f32,
    repeat_last_n: usize,
    eos_token: u32,
}

impl Model {
    pub fn load(
        weights: Vec<u8>,
        tokenizer_bytes: Vec<u8>,
        _config: Vec<u8>,  // Not used for GGUF, but keep for compatibility
    ) -> Result<Self, String> {
        // Check if SIMD is enabled
        #[cfg(feature = "simd-flash-attn")]
        ic_log!("Loading quantized Qwen3 model (Q8_0) with SIMD flash attention");

        #[cfg(not(feature = "simd-flash-attn"))]
        ic_log!("Loading quantized Qwen3 model (Q8_0) - standard mode");

        let device = Device::Cpu;

        ic_log!("Loading tokenizer...");
        let tokenizer = Tokenizer::from_bytes(&tokenizer_bytes)
            .map_err(|e| format!("Failed to load tokenizer: {}", e))?;

        // Get EOS token
        let eos_token = match tokenizer.get_vocab(true).get("<|endoftext|>") {
            Some(&token) => token,
            None => match tokenizer.get_vocab(true).get("<|im_end|>") {
                Some(&token) => token,
                None => {
                    ic_log!("⚠️  Warning: no EOS token found, using 0");
                    0
                }
            }
        };

        ic_log!("Weights size: {} bytes ({:.2} MB)",
            weights.len(),
            weights.len() as f64 / 1_048_576.0
        );

        // Load GGUF quantized model with SIMD optimizations
        let mut cursor = Cursor::new(weights);
        let content = gguf_file::Content::read(&mut cursor)
            .map_err(|e| format!("Failed to read GGUF: {}", e))?;

        ic_log!("GGUF file parsed, loading model weights...");

        // Use the new integrated API with optimizations
        let model = QuantizedQwen3::from_gguf_with_config(
            content,
            &mut cursor,
            &device,
            ComputeMode::ForceF32,  // Best for SIMD on WASM
            true,                    // use_flash_attn
            false,                   // cache_masks (not needed with SIMD)
        ).map_err(|e| format!("Failed to load model: {}", e))?;

        ic_log!("✅ Quantized model loaded successfully");

        let logits_processor = LogitsProcessor::new(299792458, None, None);

        Ok(Self {
            model,
            tokenizer,
            tokens: vec![],
            logits_processor,
            repeat_penalty: 1.,
            repeat_last_n: 64,
            eos_token,
        })
    }

    pub fn init_with_prompt(
        &mut self,
        prompt: String,
        temp: f64,
        top_p: f64,
        repeat_penalty: f32,
        repeat_last_n: usize,
        seed: u64,
    ) -> Result<String, String> {
        // Clear KV cache
        self.clear_kv_cache();

        let temp = if temp <= 0. { None } else { Some(temp) };
        let top_p = if top_p <= 0. || top_p >= 1. {
            None
        } else {
            Some(top_p)
        };

        self.logits_processor = LogitsProcessor::new(seed, temp, top_p);
        self.repeat_penalty = repeat_penalty;
        self.repeat_last_n = repeat_last_n;
        self.tokens.clear();

        let tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| format!("Tokenization error: {}", e))?
            .get_ids()
            .to_vec();

        ic_log!("Prompt encoded to {} tokens", tokens.len());

        let text = self.process(&tokens)
            .map_err(|e| format!("Processing error: {}", e))?;

        Ok(text)
    }

    pub fn next_token(&mut self) -> Result<String, String> {
        let last_token = *self.tokens.last()
            .ok_or_else(|| "No tokens generated yet".to_string())?;
        
        let text = self.process(&[last_token])
            .map_err(|e| format!("Token generation error: {}", e))?;
        
        Ok(text)
    }

    pub fn is_eos(&self) -> bool {
        self.tokens.last().map_or(false, |&t| t == self.eos_token)
    }

    pub fn get_token_count(&self) -> usize {
        self.tokens.len()
    }

    fn clear_kv_cache(&mut self) {
        for layer in &mut self.model.layers {
            layer.self_attn.kv_cache.reset();
        }
    }

    pub fn reset(&mut self) {
        self.tokens.clear();
        self.clear_kv_cache();
    }

    fn process(&mut self, tokens: &[u32]) -> candle_core::Result<String> {
        let dev = Device::Cpu;
        let input = Tensor::new(tokens, &dev)?.unsqueeze(0)?;

        // Calculate offset (position in sequence)
        let offset = self.tokens.len();

        let logits = self.model.forward(&input, offset)?;
        let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;

        // Apply repeat penalty if enabled
        let logits = if self.repeat_penalty == 1. {
            logits
        } else {
            let start_at = self.tokens.len().saturating_sub(self.repeat_last_n);
            let context = &self.tokens[start_at..];
            candle_transformers::utils::apply_repeat_penalty(&logits, self.repeat_penalty, context)?
        };

        let next_token = self.logits_processor.sample(&logits)?;
        self.tokens.push(next_token);

        let token = match self.tokenizer.decode(&[next_token], false) {
            Ok(token) => token,
            Err(e) => {
                ic_log!("Error decoding token: {:?}", e);
                "".to_string()
            }
        };

        Ok(token)
    }
}
