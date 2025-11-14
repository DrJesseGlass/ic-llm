use std::cell::RefCell;
use ic_stable_structures::{
    memory_manager::{MemoryId, MemoryManager, VirtualMemory},
    DefaultMemoryImpl, StableBTreeMap,
};
use ic_dev_kit_rs::model_server::ModelServer;

mod qwen3;
use qwen3::Qwen3Model;

type Memory = VirtualMemory<DefaultMemoryImpl>;

thread_local! {
    static MEMORY_MANAGER: RefCell<MemoryManager<DefaultMemoryImpl>> =
        RefCell::new(MemoryManager::init(DefaultMemoryImpl::default()));

    static REGISTRIES: RefCell<StableBTreeMap<String, Vec<u8>, Memory>> = RefCell::new(
        StableBTreeMap::init(MEMORY_MANAGER.with(|m| m.borrow().get(MemoryId::new(1))))
    );

    static MODEL_SERVER: ModelServer<Qwen3Model> = ModelServer::new();
}

// ═══════════════════════════════════════════════════════════════
//  Auto-Generated Endpoints (via macros)
// ═══════════════════════════════════════════════════════════════

// Generate ALL model server endpoints
ic_dev_kit_rs::generate_model_endpoints!(
    server: MODEL_SERVER,
    registry: REGISTRIES,
    weights_key: "model_weights",
    tokenizer_key: "tokenizer",
    get_tokenizer: |model| model.get_tokenizer()
);

// Generate ALL upload endpoints with storage integration
ic_dev_kit_rs::generate_upload_endpoints!(
    guard = "ic_dev_kit_rs::auth::is_authorized",
    registry = REGISTRIES
);

// ═══════════════════════════════════════════════════════════════
//  Lifecycle Hooks
// ═══════════════════════════════════════════════════════════════

#[ic_cdk::init]
async fn init() {
    ic_dev_kit_rs::auth::init_with_caller();
    ic_dev_kit_rs::telemetry::init();
    ic_dev_kit_rs::telemetry::log_info("Qwen3 canister initialized");
}

#[ic_cdk::pre_upgrade]
fn pre_upgrade() {
    let auth_bytes = ic_dev_kit_rs::auth::save_to_bytes();
    REGISTRIES.with(|r| ic_dev_kit_rs::storage::save_bytes(r, "__auth__", auth_bytes));
    ic_dev_kit_rs::telemetry::log_info("Pre-upgrade: saved auth state");
}

#[ic_cdk::post_upgrade]
fn post_upgrade() {
    let auth_bytes = REGISTRIES.with(|r| ic_dev_kit_rs::storage::load_bytes(r, "__auth__"));
    ic_dev_kit_rs::auth::init_from_saved(auth_bytes);
    ic_dev_kit_rs::telemetry::init();
    ic_dev_kit_rs::telemetry::log_info("Post-upgrade: restored auth state");
}

ic_cdk::export_candid!();