/// Type alias for representing different machine learning frameworks.
///
/// This type alias is used to specify the framework on which a machine learning model is based.
pub type ModelFramework = &'static str;

/// Constant representing the TensorFlow machine learning framework.
///
/// This constant is used to specify that a model is based on the TensorFlow framework.
pub const TENSORFLOW: ModelFramework = "tensorflow";

/// Constant representing the Torch machine learning framework.
///
/// This constant is used to specify that a model is based on the Torch framework.
pub const TORCH: ModelFramework = "torch";

/// Constant representing the PyTorch machine learning framework.
///
/// This constant is used to specify that a model is based on the PyTorch framework.
pub const PYTORCH: ModelFramework = "pytorch";

/// Constant representing the CatBoost machine learning framework.
///
/// This constant is used to specify that a model is based on the CatBoost framework.
pub const CATBOOST: ModelFramework = "catboost";

/// Constant representing the LightGBM machine learning framework.
///
/// This constant is used to specify that a model is based on the LightGBM framework.
pub const LIGHTGBM: ModelFramework = "lightgbm";

/// Constant representing the XGBoost machine learning framework.
///
/// This constant is used to specify that a model is based on the XGBoost framework.
pub const XGBOOST: ModelFramework = "xgboost";
