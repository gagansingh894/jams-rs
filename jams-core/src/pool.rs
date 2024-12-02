use crate::model::input::ModelInput;
use lazy_static::lazy_static;
use object_pool::Pool;
use std::sync::Arc;
use tokio::time;


lazy_static! {
    /// A globally accessible object pool for `ModelInput` objects.
    ///
    /// This pool manages a collection of reusable `ModelInput` objects. The pool
    /// is initialized with 500 objects, and additional objects can be added as needed.
    /// It uses `lazy_static` to ensure that the pool is only created once and is accessible globally.
    pub static ref MODEL_INPUT_POOL: Arc<Pool<ModelInput>> =
        Arc::new(Pool::new(500, ModelInput::default));
}

/// The background worker that refills the object pool when it falls below a certain threshold.
///
/// This asynchronous function runs in an infinite loop and periodically checks the size of the
/// `MODEL_INPUT_POOL`. If the pool has fewer than 400 objects, it will add 100 new `ModelInput`
/// objects to the pool. The refilling operation happens every second.
pub async fn object_pool_refiller() {
    tracing::info!("Starting object pool refill worker 🔁");

    loop {
        // Sleep for 1 second before checking the pool size again
        tokio::time::sleep(time::Duration::from_secs(1)).await;

        // Clone the pool reference to work with it
        let pool = Arc::clone(&MODEL_INPUT_POOL);

        // Check if the pool has fewer than 400 objects
        if pool.len() < 400 {
            // Attach 100 new ModelInput objects to the pool
            for _ in 0..100 {
                pool.attach(ModelInput::default())
            }
        }
    }
}