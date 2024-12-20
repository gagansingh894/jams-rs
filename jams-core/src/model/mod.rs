use crate::model::catboost::Catboost;
use crate::model::lightgbm::LightGBM;
use crate::model::predict::Predict;
use crate::model::tensorflow::Tensorflow;
use crate::model::torch::Torch;

#[cfg(feature = "xgboost")]
use crate::model::xgboost::XGBoost;

#[cfg(feature = "catboost")]
pub mod catboost;

#[cfg(feature = "lightgbm")]
pub mod lightgbm;

#[cfg(feature = "tensorflow")]
pub mod tensorflow;

#[cfg(feature = "torch")]
pub mod torch;

#[cfg(feature = "xgboost")]
mod xgboost;

// Always included modules
pub mod frameworks;
pub mod input;
pub mod output;
pub mod predict;
mod test_utils;

/// Enum representing different types of machine learning models.
pub enum Predictor {
    /// CatBoost model predictor.
    Catboost(Catboost),

    /// LightGBM model predictor.
    LightGBM(LightGBM),

    /// TensorFlow model predictor.
    Tensorflow(Tensorflow),

    /// Torch model predictor.
    Torch(Torch),

    #[cfg(feature = "xgboost")]
    /// XGBoost model predictor.
    XGBoost(XGBoost),
}
impl Predictor {
    /// Make a prediction using the appropriate machine learning model.
    ///
    /// This function will call the `predict` method of the specific model contained
    /// within the `Predictor` enum (Catboost, LightGBM, TensorFlow, Torch, or XGBoost).
    ///
    /// # Arguments
    ///
    /// * `input` - The input data for the model prediction, typically a structured
    ///   set of features or data necessary for the prediction.
    ///
    /// # Returns
    ///
    /// * `Output` - The prediction result, typically a prediction in the form of
    ///   a vector, scalar, or other model-specific type.
    ///
    /// # Errors
    ///
    /// This method will return an error if any of the models fail to perform the
    /// prediction, which may be due to issues in the underlying model, data
    /// compatibility, or other errors.
    ///
    pub fn predict(&self, input: input::ModelInput) -> anyhow::Result<output::ModelOutput> {
        match self {
            Predictor::Catboost(predictor) => predictor.predict(input),
            Predictor::LightGBM(predictor) => predictor.predict(input),
            Predictor::Tensorflow(predictor) => predictor.predict(input),
            Predictor::Torch(predictor) => predictor.predict(input),
            #[cfg(feature = "xgboost")]
            Predictor::XGBoost(predictor) => predictor.predict(input),
        }
    }
}
