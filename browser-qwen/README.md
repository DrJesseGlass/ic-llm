# Qwen3-0.6B on Internet Computer

Run a fully functional 600M parameter language model entirely in your browser, deployed on the Internet Computer blockchain.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![IC](https://img.shields.io/badge/Internet_Computer-Powered-29abe2)](https://internetcomputer.org/)
[![Rust](https://img.shields.io/badge/Rust-WASM-orange)](https://www.rust-lang.org/)
[![React](https://img.shields.io/badge/React-18-blue)](https://reactjs.org/)

## Features

- **100% Browser-Based**: No server-side inference, runs entirely in WebAssembly
- **9 tokens/sec**: Fast inference using SIMD optimizations
- **Decentralized Hosting**: Deployed on Internet Computer canisters
- **Deterministic Generation**: Reproducible outputs with seed control
- **Thinking Mode**: Optional reasoning display showing model's thought process
- **Quantized Model**: Q8 quantization for optimal speed/quality balance
- **Progressive Loading**: Smart chunked download with progress tracking

## Live Demo

[Try it here](https://your-canister-id.ic0.app) *(replace with your deployed URL)*

## Tech Stack

- **Model**: [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) (Alibaba Cloud)
- **Inference**: [Candle](https://github.com/huggingface/candle) ML framework (Rust)
- **Frontend**: React + Vite
- **Deployment**: Internet Computer (ICP)
- **Optimization**: WebAssembly + SIMD

## Prerequisites

- Node.js ≥ 16.0.0
- npm ≥ 7.0.0
- dfx (IC SDK) - [Install here](https://internetcomputer.org/docs/current/developer-docs/getting-started/install/)
- Rust + wasm-pack - [Install here](https://rustwasm.github.io/wasm-pack/installer/)
- ~700MB disk space for model files

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/qwen3-ic-browser.git
cd qwen3-ic-browser
```

### 2. Install Dependencies
```bash
npm install
cd src/frontend && npm install && cd ../..
```

### 3. Download Model Files
```bash
cd src/frontend/public/assets/wasm

# Download model weights (~610MB)
wget https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf

# Download tokenizer (~11MB)
wget https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer.json

cd ../../../../
```

### 4. Build WASM Artifacts

From DrJesseGlass's [Candle Fork](https://github.com/DrJesseGlass/candle/tree/examples/wasm/quant-qwen3) repository:
```bash
cd candle/examples/wasm/quant-qwen3
wasm-pack build --target web --release
```

Copy the generated files to your project:
```bash
cp pkg/candle_wasm_example_quant_qwen3_bg.wasm <path-to-project>/src/frontend/public/assets/wasm/
cp pkg/candle_wasm_example_quant_qwen3.js <path-to-project>/src/frontend/public/assets/wasm/
```

### 5. Deploy Locally
```bash
# Start local IC replica
dfx start --background

# Deploy canisters
dfx deploy

# Get your local URL
dfx canister id frontend --network local
# Open: http://<canister-id>.localhost:4943
```

### 6. Deploy to Mainnet
```bash
dfx deploy --network ic
```

## 🎮 Usage

### Basic Generation

1. Enter a prompt in the text box
2. Click "Generate" or press Enter
3. Watch as the model generates text token-by-token
4. View generation stats (tokens, time, speed)

### Thinking Mode

Click the "▶ Reasoning Process" dropdown to see the model's internal reasoning:
```
▼ Reasoning Process
Okay, the user is asking about X. I need to consider Y and Z...

Response:
The answer is based on...
```

### Configuration

- **Max Tokens**: Control generation length (1-500)
- **Temperature**: Modify in `useQwenModel.js` (default: 0.7)
- **Seed**: Change for deterministic/random outputs
- **Stop Generation**: Click "Stop" during generation

## Project Structure
```
qwen3-ic-browser/
├── dfx.json                    # IC canister configuration
├── src/
│   └── frontend/
│       ├── public/
│       │   ├── assets/wasm/    # Model files & WASM artifacts
│       │   └── .ic-assets.json5 # Asset canister config
│       ├── src/
│       │   ├── components/
│       │   │   ├── ChatInterface.jsx
│       │   │   └── ChatInterface.css
│       │   ├── hooks/
│       │   │   └── useQwenModel.js  # Model loading & inference
│       │   ├── main.jsx
│       │   └── index.scss
│       ├── vite.config.js
│       └── package.json
└── README.md
```

## Architecture

### Model Loading Pipeline

1. **WASM Initialization**: Load Candle runtime
2. **Chunked Download**: Stream 610MB model with progress tracking
3. **Efficient Assembly**: Combine chunks with event loop yielding
4. **Model Instantiation**: Parse GGUF format in WASM
5. **Ready State**: Model available for inference

### Inference Flow
```
User Input → Chat Template → Model.init_with_prompt() → 
Token Generation Loop → Token Filtering → UI Update
```

### Thinking Mode Architecture
```
Prompt with <think> tag → Model generates:
  <think>reasoning content</think>response content →
Parse and split → Display separately
```

## Performance

- **Initial Load**: ~30-60s (one-time, cached after)
- **Inference Speed**: ~9 tokens/second
- **Model Size**: 610MB (Q8 quantization)
- **Memory Usage**: ~800MB browser heap

### Optimization Tips

1. **Service Worker Caching**: Add for instant subsequent loads
2. **Q4 Quantization**: Use smaller model (~380MB) for faster load
3. **CDN Hosting**: Host model on external CDN, reduce IC storage costs

## Contributing

Contributions welcome! Areas for improvement:

- [ ] Multi-turn conversation support
- [ ] System prompt customization UI
- [ ] Model selection (different Qwen variants)
- [ ] Export conversation history
- [ ] Mobile optimization
- [ ] Service worker implementation
- [ ] Performance benchmarking suite

### Development Setup
```bash
# Install dependencies
npm install

# Start local development (with hot reload)
cd src/frontend
npm run dev

# Build for production
npm run build
```

## License

MIT License - see [LICENSE](LICENSE) file for details

## Acknowledgments

- **Qwen Team** (Alibaba Cloud) - Base model
- **Unsloth** - Quantized GGUF model
- **Hugging Face** - Candle ML framework
- **DFINITY** - Internet Computer platform
- **Anthropic** - Development assistance

## 📚 Resources

- [Qwen3 Model Card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Candle Documentation](https://github.com/huggingface/candle)
- [Internet Computer Docs](https://internetcomputer.org/docs)
- [WASM-Bindgen Guide](https://rustwasm.github.io/docs/wasm-bindgen/)

## 🐛 Known Issues

- **Firefox**: May require COOP/COEP headers adjustment
- **Safari**: Limited SharedArrayBuffer support, may have performance issues
- **Mobile**: Large model size may cause memory issues on lower-end devices

## 💬 Contact

- GitHub Issues: [Project Issues](https://github.com/yourusername/qwen3-ic-browser/issues)
- Twitter: [@yourhandle](https://twitter.com/yourhandle)

---

**Star ⭐ this repo if you find it useful!**
