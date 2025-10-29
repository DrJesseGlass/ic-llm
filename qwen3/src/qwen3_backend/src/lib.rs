use std::cell::RefCell;
use candid::{CandidType, Decode, Encode};
use ic_stable_structures::{
    memory_manager::{MemoryId, MemoryManager, VirtualMemory},
    DefaultMemoryImpl, StableBTreeMap,
};

pub mod storage;
pub mod candle;

// Re-export Qwen3 types for Candid interface
pub use candle::{
    GenerationConfig,
    InferenceRequest,
    InferenceResponse,
    ModelInfo,
    EmptyResult,
    generate,
    setup_model,
    reset_generation,
    is_model_loaded,
    get_model_info,
};

// Re-export storage functions for Candid
pub use storage::{
    // Sequential upload
    append_chunk,
    buffer_size,
    clear_buffer,

    // Parallel upload
    append_parallel_chunk,
    parallel_chunk_count,
    parallel_chunk_ids,
    parallel_buffer_size,
    parallel_chunks_complete,
    consolidate_parallel_chunks,
    clear_parallel_chunks,
    remove_parallel_chunk,

    // Stable storage
    save_to_stable,
    save_parallel_to_stable,
    load_from_stable,
    get_data,
    get_stable_data,
    storage_status,
};

type Memory = VirtualMemory<DefaultMemoryImpl>;

thread_local! {
    pub static MEMORY_MANAGER: RefCell<MemoryManager<DefaultMemoryImpl>> =
        RefCell::new(MemoryManager::init(DefaultMemoryImpl::default()));

    pub static REGISTRIES: RefCell<StableBTreeMap<String, Vec<u8>, Memory>> = RefCell::new(
        StableBTreeMap::init(
            MEMORY_MANAGER.with(|m| m.borrow().get(MemoryId::new(1))),
        )
    );
}

// ─────────────────────────────────────────────────────
//  Simplified Helper Functions
// ─────────────────────────────────────────────────────

/// Save any CandidType directly as Vec<u8>
pub fn save_data_to_stable<T: CandidType>(key: &str, data: &T) {
    match Encode!(data) {
        Ok(serialized_bytes) => {
            REGISTRIES.with(|map| {
                map.borrow_mut().insert(key.to_string(), serialized_bytes);
            });
        }
        Err(e) => {
            eprintln!("Failed to serialize data for key {}: {:?}", key, e);
        }
    }
}

/// Load any CandidType directly from Vec<u8>
pub fn load_data_from_stable<T>(key: &str) -> Option<T>
where
    T: for<'de> candid::Deserialize<'de> + CandidType,
{
    REGISTRIES.with(|map| {
        if let Some(serialized_bytes) = map.borrow().get(&key.to_string()) {
            match Decode!(&serialized_bytes, T) {
                Ok(data) => Some(data),
                Err(e) => {
                    eprintln!("Failed to deserialize data for key {}: {:?}", key, e);
                    None
                }
            }
        } else {
            None
        }
    })
}

/// Save raw bytes directly
pub fn save_bytes_to_stable(key: &str, bytes: Vec<u8>) {
    REGISTRIES.with(|map| {
        map.borrow_mut().insert(key.to_string(), bytes);
    });
}

/// Load raw bytes directly
pub fn load_bytes_from_stable(key: &str) -> Option<Vec<u8>> {
    REGISTRIES.with(|map| {
        map.borrow().get(&key.to_string())
    })
}

#[ic_cdk::init]
fn init() {
    ic_cdk::println!("Qwen3 canister initialized");
}

#[ic_cdk::pre_upgrade]
fn pre_upgrade() {
    ic_cdk::println!("Pre-upgrade: Stable storage will persist");
}

#[ic_cdk::post_upgrade]
fn post_upgrade() {
    ic_cdk::println!("Post-upgrade: Canister upgraded");
}

ic_cdk::export_candid!();