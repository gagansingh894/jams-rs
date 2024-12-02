use crate::model::input::ModelInput;
use lazy_static::lazy_static;
use object_pool::Pool;
use std::sync::Arc;
use tokio::time;

lazy_static! {
    pub static ref MODEL_INPUT_POOL: Arc<Pool<ModelInput>> =
        Arc::new(Pool::new(1000, ModelInput::default));
}

pub async fn background_refill_object_pool() {
    tracing::info!("Starting object pool refill worker 🔁");
    loop {
        tokio::time::sleep(time::Duration::from_secs(1)).await;
        let pool = Arc::clone(&MODEL_INPUT_POOL);
        if pool.len() < 500 {
            for _ in 0..100 {
                pool.attach(ModelInput::default())
            }
        }
    }
}
