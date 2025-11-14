//! Qwen3 model - only Qwen3-specific logic

use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_qwen3::ModelWeights as QuantizedQwen3;
use ::tokenizers::Tokenizer;  // Use :: to explicitly refer to the external crate

// Import from ic-dev-kit-rs
use ic_dev_kit_rs::candle::*;
use ic_dev_kit_rs::text_generation::*;

pub struct Qwen3Model {
    model: QuantizedQwen3,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    tokens: Vec<u32>,
    repeat_penalty: f32,
    repeat_last_n: usize,
    eos_token: u32,
}

pub struct Qwen3Tokenizer(Tokenizer);

impl TokenizerHandle for Qwen3Tokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        self.0.encode(text, true)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| format!("Encode error: {}", e))
    }

    fn decode(&self, tokens: &[u32]) -> Result<String, String> {
        self.0.decode(tokens, false).map_err(|e| format!("Decode error: {}", e))
    }

    fn vocab_size(&self) -> usize {
        self.0.get_vocab_size(true)
    }
}

impl CandleModel for Qwen3Model {
    fn load(weights: Vec<u8>, config: Option<Vec<u8>>) -> Result<Self, String> {
        let tokenizer_bytes = config.ok_or("Tokenizer required")?;
        let tokenizer = Tokenizer::from_bytes(&tokenizer_bytes)
            .map_err(|e| format!("Failed to load tokenizer: {}", e))?;

        // Use helpers from ic-dev-kit - note: this is the text_generation::tokenizers module
        let eos_token = tokenizers::find_eos_token(&tokenizer);
        let (content, mut cursor) = gguf::load_content(weights)?;
        let device = gguf::cpu_device();

        let model = QuantizedQwen3::from_gguf(content, &mut cursor, &device)
            .map_err(|e| format!("Failed to load model: {}", e))?;

        Ok(Self {
            model,
            tokenizer,
            tokens: vec![],
            logits_processor: LogitsProcessor::new(299792458, None, None),
            repeat_penalty: 1.,
            repeat_last_n: 64,
            eos_token,
        })
    }

    fn metadata(&self) -> ModelMetadata {
        ModelMetadata {
            name: "Qwen3".to_string(),
            version: "0.5B".to_string(),
            architecture: "Qwen3 (Quantized GGUF)".to_string(),
            parameters: 500_000_000,
            context_length: Some(8192),
        }
    }

    fn reset(&mut self) {
        self.tokens.clear();
    }
}

impl AutoregressiveModel for Qwen3Model {
    fn init_generation(
        &mut self,
        prompt: String,
        tokenizer: &dyn TokenizerHandle,
        config: &GenerationConfig,
    ) -> Result<String, String> {
        let temp = if config.temperature <= 0. { None } else { Some(config.temperature) };
        let top_p = if config.top_p <= 0. || config.top_p >= 1. { None } else { Some(config.top_p) };

        self.logits_processor = LogitsProcessor::new(config.seed, temp, top_p);
        self.repeat_penalty = config.repeat_penalty;
        self.repeat_last_n = config.repeat_last_n;
        self.tokens.clear();

        let tokens = tokenizer.encode(&prompt)?;
        self.process(&tokens).map_err(|e| e.to_string())
    }

    fn generate_next_token(&mut self, _tokenizer: &dyn TokenizerHandle) -> Result<String, String> {
        let last_token = *self.tokens.last().ok_or("No tokens generated")?;
        self.process(&[last_token]).map_err(|e| e.to_string())
    }

    fn is_generation_complete(&self) -> bool {
        self.tokens.last().map_or(false, |&t| t == self.eos_token)
    }

    fn generated_token_count(&self) -> usize {
        self.tokens.len()
    }
}

impl Qwen3Model {
    pub fn get_tokenizer(&self) -> Box<dyn TokenizerHandle> {
        Box::new(Qwen3Tokenizer(self.tokenizer.clone()))
    }

    fn process(&mut self, tokens: &[u32]) -> candle_core::Result<String> {
        use candle_core::{DType, Device, Tensor};

        let input = Tensor::new(tokens, &Device::Cpu)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, self.tokens.len())?.squeeze(0)?.to_dtype(DType::F32)?;

        let logits = if self.repeat_penalty != 1. {
            let start = self.tokens.len().saturating_sub(self.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(&logits, self.repeat_penalty, &self.tokens[start..])?
        } else {
            logits
        };

        let next_token = self.logits_processor.sample(&logits)?;
        self.tokens.push(next_token);

        self.tokenizer.decode(&[next_token], false)
            .map_err(|e| candle_core::Error::Msg(format!("{:?}", e)))
    }
}