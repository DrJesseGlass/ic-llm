# ic-llm

Two experiments running an LLM (Qwen3-0.6B, Q4_K GGUF) on the Internet Computer, and the
benchmark evidence for why neither on-chain inference nor browser-WASM inference is the
right primary architecture.

- [`backend-qwen3/`](backend-qwen3/) -- the model running **fully on-chain** in a Rust
  canister (candle), with a persisted [canbench](https://github.com/dfinity/canbench)
  baseline in [`backend-qwen3/canbench_results.yml`](backend-qwen3/canbench_results.yml).
- [`browser-qwen/`](browser-qwen/) -- the model running **in the browser** via WASM
  (candle fork with SIMD + rayon threading), served as static assets from an IC canister.
  ~9 tokens/sec.

## TL;DR

**On-chain generative inference is not viable, and this is measured, not vibes.** Decoding
one token of a 0.6B-parameter Q4 model costs ~7 billion instructions. At IC fee rates that
is roughly **$3,700 per 1M output tokens** at ~0.3 tokens/sec -- a thousand times frontier-API
prices for a model that costs pennies to serve anywhere else. This is the cost of replicated
deterministic execution, not an engineering gap.

**Browser WASM works but is superseded by WebGPU.** 9 tok/s WASM vs 40-150 tok/s for the
same model on WebGPU (WebLLM/MLC-class stacks), with prefill faster by a larger factor
still -- and WebGPU makes 4B-8B models usable in-browser, which changes output quality
categorically.

**What survives: the IC as host.** In the browser approach the IC's only job is serving
static assets (the 326MB GGUF and the app shell). That role is identical whether the
compute is WASM or WebGPU. What died is compute-on-IC, not hosting-on-IC.

The defensible architecture: **WebGPU-first in the browser, IC asset canister as the
decentralized host, WASM as the no-GPU fallback, and on-chain inference reserved for
sub-second trusted micro-inference** (see below).

## Backend benchmarks, in real units

Measured with canbench on the wasm32 canister build. Conversions assume the standard
13-node app-subnet fee schedule (0.4 cycles/instruction, ~$1.33 per 1T cycles), the 40B
instruction ceiling per update call (DTS), and ~2B instructions executed per ~1s round.

| Metric | Measured | Implication |
|---|---|---|
| Decode | ~6.9-7.2B instr/token | ~$0.0037/token -> **~$3,700 / 1M output tokens**; ~3.5s/token wall clock |
| Prefill | ~5.1B instr/prompt token | **~$2,700 / 1M input tokens** |
| 200-token prompt | 1.03T instructions | ~26 chained max-size update calls, **~8 minutes** wall clock, ~$0.55 |
| Model load (GGUF parse) | 9.6B instructions | must stay resident in heap; reload on upgrade costs a full call |
| One update call (40B cap) | -- | fits ~5 decoded tokens, total |
| One decode (7B) | -- | exceeds the 5B query-call limit -- no free-query path exists |
| Tokenize (short/med/long) | 46-102M instructions | noise; tokenization is never the problem |

Two properties of the data worth trusting:

- **Decode cost is flat in context depth** -- 6.89B -> 7.15B instructions from context 16 to
  128 (~4%). At these depths the matmuls dominate, not attention, so the per-token numbers
  extrapolate linearly.
- **Prefill and decode are isolated** -- KV-cache priming runs outside the measured region
  (see `backend-qwen3/src/qwen3_backend/src/canbench_benchmarks.rs`), so the per-stage
  costs are clean.

The conclusion is consistent with DFINITY's own direction: their "LLM canister" routes
inference through off-chain AI workers precisely because on-chain execution can't carry it.

### The one thing on-chain inference still buys: trust

Browser inference produces outputs the chain cannot verify -- the client can lie. If a use
case requires **consensus-verified inference plus autonomous action** (timers, holding
assets, inter-canister calls), on-chain is the only option, and the benchmarks now price
that trust precisely: ~$4 per 1K tokens. For a 10-20 token structured decision that's ~5
cents and tolerable; for chat it's absurd.

Non-generative models remain comfortably viable on-chain: an embedding model, classifier,
or reranker fits in a single update call at fractions of a cent. So the honest framing is
*"backend is only for short, high-trust outputs"* -- not "backend can serve LLMs."

## Frontend: WASM vs WebGPU

The WASM build (candle fork: multithread, relaxed SIMD, lazy Q4K dequant, streaming load)
hits ~9 tok/s on Qwen3-0.6B. WebGPU runs the same model at 40-150 tok/s on a discrete GPU,
with prefill latency -- the part users actually feel on long prompts -- improved by an even
larger factor. More importantly, WebGPU makes 4B-8B models practical in-browser.

The WASM path is demoted, not deleted:

- WebGPU coverage (Chrome/Edge, Safari 26, Firefox on Windows) is good but not universal --
  older hardware, driver blocklists, some Android. The standard architecture
  (e.g. transformers.js) is WebGPU-first with WASM fallback, and the candle-fork work here
  is exactly the fallback tech. It's finished and it works.
- For the WebGPU path, use an MLC/WebLLM-class stack and its weight format rather than the
  candle GGUF pipeline -- candle's WebGPU backend is not the mature option.

## Reproducing the benchmarks

```bash
cd backend-qwen3
# model + tokenizer into benchmark-assets/ (gitignored):
huggingface-cli download DrJesseGlass/Qwen3-0.6B-Q4_K Qwen3-0.6B-allq4k-f16src.gguf \
  --local-dir src/qwen3_backend/benchmark-assets
huggingface-cli download Qwen/Qwen3-0.6B tokenizer.json \
  --local-dir src/qwen3_backend/benchmark-assets
canbench   # compares against the persisted baseline in canbench_results.yml
```

See each subproject's README for deploy instructions.
