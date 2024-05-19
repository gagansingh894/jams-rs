use crate::model::predictor::{ModelInput, Predictor};
use tensorflow::{Graph, SavedModelBundle, SessionOptions};

pub struct Tensorflow {
    bundle: SavedModelBundle,
}

impl Tensorflow {
    pub fn load(model_dir: &str) -> anyhow::Result<Self> {
        const MODEL_TAG: &str = "serve";
        let mut graph = Graph::new();
        let bundle =
            SavedModelBundle::load(&SessionOptions::new(), &[MODEL_TAG], &mut graph, model_dir)
                .expect("failed to load tensorflow savedmodel");
        Ok(Tensorflow { bundle })
    }
}

impl Predictor for Tensorflow {
    fn predict(input: ModelInput) {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn successfully_load_tensorflow_regression_model() {
        let model_dir = "tests/model_artefacts/autompg_tensorflow";
        let model = Tensorflow::load(model_dir);

        // assert the result is Ok
        assert!(model.is_ok())
    }

    #[test]
    fn successfully_load_tensorflow_multi_classification_model() {
        let model_dir = "tests/model_artefacts/mnist_tensorflow";
        let model = Tensorflow::load(model_dir);

        // assert the result is Ok
        assert!(model.is_ok())
    }
}
