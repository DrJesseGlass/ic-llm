#!/bin/bash
set -ex  # Changed from -e to -ex for verbose output

echo "=== Building qwen3_backend with SIMD ==="
pwd
ls -la src/qwen3_backend/

# Check package name
echo "Checking Cargo.toml package name..."
grep "^name" src/qwen3_backend/Cargo.toml

# Build with SIMD flags
RUSTFLAGS="-C target-feature=+simd128" cargo build \
  --target wasm32-unknown-unknown \
  --release \
  --features simd-flash-attn \
  --package qwen3-backend

# Check if wasm was created
echo "=== Checking for wasm file ==="
ls -lh target/wasm32-unknown-unknown/release/*.wasm

echo "=== Build complete! ==="