// src/storage.rs
//! Storage module for Qwen3: chunked uploads + stable storage

use std::cell::RefCell;
use std::collections::HashMap;
use crate::REGISTRIES;

// Single buffer in heap for sequential uploads
thread_local! {
    static BUFFER: RefCell<Vec<u8>> = RefCell::new(Vec::new());
    static BUFFER_MAP: RefCell<HashMap<u32, Vec<u8>>> = RefCell::new(HashMap::new());
}

// ─────────────────────────────────────────────────────
//  Sequential Upload (Original Pattern)
// ─────────────────────────────────────────────────────

/// Append chunk to the single heap buffer
#[ic_cdk::update]
pub fn append_chunk(chunk: Vec<u8>) {
    BUFFER.with(|buffer| {
        buffer.borrow_mut().extend(chunk);
    });
}

/// Get current buffer size
#[ic_cdk::query]
pub fn buffer_size() -> usize {
    BUFFER.with(|buffer| buffer.borrow().len())
}

/// Clear the heap buffer
#[ic_cdk::update]
pub fn clear_buffer() {
    BUFFER.with(|buffer| {
        buffer.borrow_mut().clear();
    });
}

// ─────────────────────────────────────────────────────
//  Parallel Upload (Faster for Large Models)
// ─────────────────────────────────────────────────────

/// Append chunk with ID for parallel uploads
#[ic_cdk::update]
pub fn append_parallel_chunk(chunk_id: u32, chunk: Vec<u8>) {
    BUFFER_MAP.with(|buffer_map| {
        buffer_map.borrow_mut().insert(chunk_id, chunk);
    });
}

/// Get number of chunks in the parallel buffer
#[ic_cdk::query]
pub fn parallel_chunk_count() -> usize {
    BUFFER_MAP.with(|buffer_map| buffer_map.borrow().len())
}

/// Get list of chunk IDs currently in the parallel buffer
#[ic_cdk::query]
pub fn parallel_chunk_ids() -> Vec<u32> {
    BUFFER_MAP.with(|buffer_map| {
        let mut ids: Vec<u32> = buffer_map.borrow().keys().copied().collect();
        ids.sort();
        ids
    })
}

/// Get total size of all chunks in parallel buffer
#[ic_cdk::query]
pub fn parallel_buffer_size() -> usize {
    BUFFER_MAP.with(|buffer_map| {
        buffer_map.borrow().values().map(|chunk| chunk.len()).sum()
    })
}

/// Check if all chunks from 0 to expected_count-1 are present
#[ic_cdk::query]
pub fn parallel_chunks_complete(expected_count: u32) -> bool {
    BUFFER_MAP.with(|buffer_map| {
        let buffer_map = buffer_map.borrow();
        if buffer_map.len() != expected_count as usize {
            return false;
        }

        // Check consecutive chunks
        for i in 0..expected_count {
            if !buffer_map.contains_key(&i) {
                return false;
            }
        }
        true
    })
}

/// Consolidate parallel chunks into main buffer (in order)
#[ic_cdk::update]
pub fn consolidate_parallel_chunks() -> Result<usize, String> {
    let (chunk_data, total_size) = BUFFER_MAP.with(|buffer_map| {
        let mut buffer_map = buffer_map.borrow_mut();

        if buffer_map.is_empty() {
            return (Vec::new(), 0);
        }

        // Sort and collect data
        let mut sorted_ids: Vec<u32> = buffer_map.keys().copied().collect();
        sorted_ids.sort();

        let mut consolidated_data = Vec::new();
        let mut total_size = 0;

        for chunk_id in sorted_ids {
            if let Some(chunk) = buffer_map.remove(&chunk_id) {
                total_size += chunk.len();
                consolidated_data.extend(chunk);
            }
        }

        buffer_map.clear();
        (consolidated_data, total_size)
    });

    if chunk_data.is_empty() {
        return Err("No parallel chunks to consolidate".to_string());
    }

    // Move to main buffer
    BUFFER.with(|buffer| {
        let mut buffer = buffer.borrow_mut();
        buffer.clear();
        buffer.extend(chunk_data);
    });

    Ok(total_size)
}

/// Clear all parallel chunks
#[ic_cdk::update]
pub fn clear_parallel_chunks() {
    BUFFER_MAP.with(|buffer_map| {
        buffer_map.borrow_mut().clear();
    });
}

/// Remove specific chunk (for retry scenarios)
#[ic_cdk::update]
pub fn remove_parallel_chunk(chunk_id: u32) -> bool {
    BUFFER_MAP.with(|buffer_map| {
        buffer_map.borrow_mut().remove(&chunk_id).is_some()
    })
}

// ─────────────────────────────────────────────────────
//  Stable Storage
// ─────────────────────────────────────────────────────

/// Save buffer to stable storage with key
#[ic_cdk::update]
pub fn save_to_stable(key: String) -> Result<(), String> {
    let data = BUFFER.with(|buffer| {
        let mut buffer = buffer.borrow_mut();
        std::mem::take(&mut *buffer)
    });

    if data.is_empty() {
        return Err(format!("No data in buffer for key: {}", key));
    }

    REGISTRIES.with(|map| {
        map.borrow_mut().insert(key, data);
    });

    Ok(())
}

/// Save parallel chunks directly to stable storage
#[ic_cdk::update]
pub fn save_parallel_to_stable(key: String) -> Result<usize, String> {
    let consolidated_data = BUFFER_MAP.with(|buffer_map| {
        let mut buffer_map = buffer_map.borrow_mut();

        if buffer_map.is_empty() {
            return Vec::new();
        }

        let mut sorted_ids: Vec<u32> = buffer_map.keys().copied().collect();
        sorted_ids.sort();

        let mut consolidated_data = Vec::new();

        for chunk_id in sorted_ids {
            if let Some(chunk) = buffer_map.remove(&chunk_id) {
                consolidated_data.extend(chunk);
            }
        }

        buffer_map.clear();
        consolidated_data
    });

    if consolidated_data.is_empty() {
        return Err(format!("No parallel chunks to save for key: {}", key));
    }

    let data_size = consolidated_data.len();

    REGISTRIES.with(|map| {
        map.borrow_mut().insert(key, consolidated_data);
    });

    Ok(data_size)
}

/// Load from stable storage to buffer
#[ic_cdk::update]
pub fn load_from_stable(key: String) -> Result<(), String> {
    REGISTRIES.with(|map| {
        if let Some(data) = map.borrow().get(&key) {
            BUFFER.with(|buffer| {
                buffer.borrow_mut().clone_from(&data);
            });
            Ok(())
        } else {
            Err(format!("No data found in stable storage for key: {}", key))
        }
    })
}

/// Get buffered data (consumes buffer)
#[ic_cdk::update]
pub fn get_data() -> Vec<u8> {
    BUFFER.with(|buffer| {
        let mut buffer = buffer.borrow_mut();
        std::mem::take(&mut *buffer)
    })
}

/// Get data directly from stable storage
#[ic_cdk::query]
pub fn get_stable_data(key: String) -> Result<Vec<u8>, String> {
    REGISTRIES.with(|map| {
        map.borrow().get(&key)
            .ok_or_else(|| format!("No data found in stable storage for key: {}", key))
    })
}

// ─────────────────────────────────────────────────────
//  Monitoring
// ─────────────────────────────────────────────────────

/// Get storage status summary
#[ic_cdk::query]
pub fn storage_status() -> String {
    let buffer_size = BUFFER.with(|buffer| buffer.borrow().len());

    let (chunk_count, parallel_size, chunk_ids) = BUFFER_MAP.with(|buffer_map| {
        let buffer_map = buffer_map.borrow();
        let count = buffer_map.len();
        let size = buffer_map.values().map(|chunk| chunk.len()).sum::<usize>();
        let mut ids: Vec<u32> = buffer_map.keys().copied().collect();
        ids.sort();
        (count, size, ids)
    });

    let stable_keys = REGISTRIES.with(|map| {
        map.borrow()
            .iter()
            .map(|entry| {
                format!("{}: {} bytes", entry.key(), entry.value().len())
            })
            .collect::<Vec<_>>()
    });

    format!(
        "Buffer: {} bytes\n\
         Parallel chunks: {} chunks, {} bytes total\n\
         Chunk IDs: {:?}\n\
         Stable storage: [{}]",
        buffer_size,
        chunk_count,
        parallel_size,
        chunk_ids,
        stable_keys.join(", ")
    )
}