#!/usr/bin/env python3
"""
Upload Qwen3 model to IC canister in chunks.
Handles large GGUF files by splitting into manageable pieces.
"""
import os
import sys
import subprocess
import json
from pathlib import Path

# HuggingFace cache locations (same as your serve.py)
HOME = Path.home()

# GGUF model from unsloth
GGUF_BASE = HOME / '.cache/huggingface/hub/models--unsloth--Qwen3-0.6B-GGUF'
gguf_snapshots = list((GGUF_BASE / 'snapshots').glob('*'))
if not gguf_snapshots:
    print(f"Error: No snapshots found in {GGUF_BASE / 'snapshots'}", file=sys.stderr)
    print("Run: huggingface-cli download unsloth/Qwen3-0.6B-GGUF Qwen3-0.6B-Q8_0.gguf")
    sys.exit(1)
GGUF_SNAPSHOT = gguf_snapshots[0]

# Tokenizer and config from original Qwen3 repo
QWEN_BASE = HOME / '.cache/huggingface/hub/models--Qwen--Qwen3-0.6B'
qwen_snapshots = list((QWEN_BASE / 'snapshots').glob('*'))
if not qwen_snapshots:
    print(f"Error: No snapshots found in {QWEN_BASE / 'snapshots'}", file=sys.stderr)
    print("Run: huggingface-cli download Qwen/Qwen3-0.6B tokenizer.json config.json")
    sys.exit(1)
QWEN_SNAPSHOT = qwen_snapshots[0]

# Choose model file
GGUF_FILE = 'Qwen3-0.6B-Q8_0.gguf'  # ~600MB
# GGUF_FILE = 'Qwen3-0.6B-Q4_K_M.gguf'  # ~400MB, faster upload

WEIGHTS_PATH = GGUF_SNAPSHOT / GGUF_FILE
TOKENIZER_PATH = QWEN_SNAPSHOT / 'tokenizer.json'
CONFIG_PATH = QWEN_SNAPSHOT / 'config.json'

# IC settings
CHUNK_SIZE = 1_900_000  # ~1.9MB per chunk (IC message limit is 2MB)
CANISTER_NAME = "qwen3_backend"


def run_dfx_command(command: list[str], input_data: bytes = None) -> tuple[bool, str]:
    """Run a dfx command and return success status and output."""
    try:
        result = subprocess.run(
            command,
            input=input_data,
            capture_output=True,
            timeout=300,  # 5 minute timeout
        )
        output = result.stdout.decode('utf-8', errors='ignore')
        if result.returncode != 0:
            error = result.stderr.decode('utf-8', errors='ignore')
            print(f"Error: {error}", file=sys.stderr)
            return False, error
        return True, output
    except subprocess.TimeoutExpired:
        print("Command timed out", file=sys.stderr)
        return False, "Timeout"
    except Exception as e:
        print(f"Exception: {e}", file=sys.stderr)
        return False, str(e)


def upload_file_in_chunks(file_path: Path, method_name: str) -> bool:
    """Upload a file to the canister in chunks."""
    file_size = file_path.stat().st_size
    print(f"\nUploading {file_path.name} ({file_size / 1_048_576:.2f} MB)...")

    with open(file_path, 'rb') as f:
        data = f.read()

    total_chunks = (len(data) + CHUNK_SIZE - 1) // CHUNK_SIZE

    for i in range(total_chunks):
        start = i * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, len(data))
        chunk = data[start:end]

        print(f"  Uploading chunk {i + 1}/{total_chunks} ({len(chunk)} bytes)...", end=' ')

        # Convert chunk to hex for dfx
        chunk_hex = chunk.hex()

        # Build dfx command based on method
        if method_name == "upload_weights_chunk":
            cmd = [
                'dfx', 'canister', 'call', CANISTER_NAME, method_name,
                f'(blob "{chunk_hex}", {i} : nat, {total_chunks} : nat)'
            ]
        else:
            cmd = [
                'dfx', 'canister', 'call', CANISTER_NAME, method_name,
                f'(blob "{chunk_hex}")'
            ]

        success, output = run_dfx_command(cmd)

        if not success:
            print("❌ Failed")
            return False

        print("✅")

        # Show progress from canister
        if output.strip():
            # Parse the candid response
            try:
                # Extract text from candid response format
                if '"' in output:
                    msg = output.split('"')[1]
                    print(f"    Canister: {msg}")
            except:
                pass

    return True


def main():
    # Verify files exist
    for path in [WEIGHTS_PATH, TOKENIZER_PATH, CONFIG_PATH]:
        if not path.exists():
            print(f"❌ Error: {path.name} not found at {path}", file=sys.stderr)
            sys.exit(1)
        print(f"✅ Found: {path.name}")

    print(f"\n📦 Upload Configuration:")
    print(f"  Canister: {CANISTER_NAME}")
    print(f"  Chunk size: {CHUNK_SIZE / 1_048_576:.2f} MB")
    print(f"  Model: {GGUF_FILE}")
    print(f"  Total size: {WEIGHTS_PATH.stat().st_size / 1_048_576:.2f} MB")

    # Check if canister is deployed
    print("\n🔍 Checking canister status...")
    success, output = run_dfx_command(['dfx', 'canister', 'id', CANISTER_NAME])
    if not success:
        print("❌ Canister not found. Deploy with: dfx deploy")
        sys.exit(1)

    canister_id = output.strip()
    print(f"✅ Canister found: {canister_id}")

    # Upload tokenizer (small file, single upload)
    print("\n📤 Uploading tokenizer...")
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer_data = f.read()
    tokenizer_hex = tokenizer_data.hex()

    success, output = run_dfx_command([
        'dfx', 'canister', 'call', CANISTER_NAME, 'upload_tokenizer',
        f'(blob "{tokenizer_hex}")'
    ])

    if not success:
        print("❌ Failed to upload tokenizer")
        sys.exit(1)
    print("✅ Tokenizer uploaded")

    # Upload config (small file, single upload)
    print("\n📤 Uploading config...")
    with open(CONFIG_PATH, 'rb') as f:
        config_data = f.read()
    config_hex = config_data.hex()

    success, output = run_dfx_command([
        'dfx', 'canister', 'call', CANISTER_NAME, 'upload_config',
        f'(blob "{config_hex}")'
    ])

    if not success:
        print("❌ Failed to upload config")
        sys.exit(1)
    print("✅ Config uploaded")

    # Upload model weights in chunks
    if not upload_file_in_chunks(WEIGHTS_PATH, "upload_weights_chunk"):
        print("\n❌ Failed to upload model weights")
        sys.exit(1)

    print("All files uploaded successfully!")

    # Initialize model
    print("Initializing model...")
    success, output = run_dfx_command([
        'dfx', 'canister', 'call', CANISTER_NAME, 'initialize_model'
    ])

    if not success:
        print("❌ Failed to initialize model")
        sys.exit(1)

    print("✅ Model initialized!")
    print(output)

    # Get model info
    print("\n📊 Model Info:")
    success, output = run_dfx_command([
        'dfx', 'canister', 'call', CANISTER_NAME, 'get_model_info'
    ])

    if success:
        print(output)

    print("\n🎉 Setup complete! You can now run inference.")
    print(f"\nTest with:")
    print(f'  dfx canister call {CANISTER_NAME} init_with_prompt \'(record {{ prompt = "Hello"; config = null }})\'')


if __name__ == '__main__':
    main()