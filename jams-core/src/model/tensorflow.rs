use crate::model::predict::{FeatureName, ModelInput, Output, Predict};
use std::collections::HashMap;
use tensorflow::{
    DataType, FetchToken, Graph, Operation, SavedModelBundle, SessionOptions, SessionRunArgs,
    SignatureDef, Tensor, DEFAULT_SERVING_SIGNATURE_DEF_KEY,
};
use crate::FEATURE_NAMES_CAPACITY;

// Arbitrary high number but in reality, the number is far less
const MAX_OUTPUT_NODES_SUPPORTED: usize = 250;

/// Struct representing the input tensors for a TensorFlow model.
///
/// This struct encapsulates tensors grouped by their data type (`i32`, `f32`, `String`),
/// along with corresponding TensorFlow operations.
struct TensorflowModelInput {
    /// Integer tensors along with their corresponding TensorFlow operations.
    pub int_tensors: Vec<(Operation, Tensor<i32>)>,
    /// Float tensors along with their corresponding TensorFlow operations.
    pub float_tensors: Vec<(Operation, Tensor<f32>)>,
    /// String tensors along with their corresponding TensorFlow operations.
    pub string_tensors: Vec<(Operation, Tensor<String>)>,
}

impl TensorflowModelInput {
    /// Parses a `ModelInput` into a `TensorflowModelInput` based on the TensorFlow model's signature definition and graph.
    ///
    /// # Arguments
    /// * `input` - The `ModelInput` containing the input data.
    /// * `signature_def` - The TensorFlow `SignatureDef` that describes the model's input signature.
    /// * `graph` - The TensorFlow `Graph` containing the model.
    ///
    /// # Returns
    /// * `Ok(TensorflowModelInput)` - If parsing was successful.
    /// * `Err(anyhow::Error)` - If there was an error during parsing.
    #[tracing::instrument(skip(input, signature_def, graph))]
    pub fn parse(
        input: ModelInput,
        signature_def: &SignatureDef,
        graph: &Graph,
    ) -> anyhow::Result<Self> {
        let signature_def_num_values = signature_def.inputs().len();
        if signature_def_num_values == 0 {
            tracing::error!("Model graph has no inputs");
            anyhow::bail!("Model graph has no inputs");
        } else if signature_def_num_values == 1 {
            parse_sequential(input, signature_def, graph)
        } else {
            parse_functional(input, signature_def, graph)
        }
    }
}

/// Parses a `ModelInput` into a `TensorflowModelInput` when the TensorFlow model has a single input signature.
///
/// # Arguments
/// * `model_input` - The `ModelInput` containing the input data.
/// * `signature_def` - The TensorFlow `SignatureDef` that describes the model's input signature.
/// * `graph` - The TensorFlow `Graph` containing the model.
///
/// # Returns
/// * `Ok(TensorflowModelInput)` - If parsing was successful.
/// * `Err(anyhow::Error)` - If there was an error during parsing.
#[tracing::instrument(skip(model_input, signature_def, graph))]
fn parse_sequential(
    model_input: ModelInput,
    signature_def: &SignatureDef,
    graph: &Graph,
) -> anyhow::Result<TensorflowModelInput> {
    let mut int_tensors: Vec<(Operation, Tensor<i32>)> = Vec::with_capacity(FEATURE_NAMES_CAPACITY);
    let mut float_tensors: Vec<(Operation, Tensor<f32>)> = Vec::with_capacity(FEATURE_NAMES_CAPACITY * 2);
    let mut string_tensors: Vec<(Operation, Tensor<String>)> = Vec::with_capacity(FEATURE_NAMES_CAPACITY);

    // Create tensors
    for input in signature_def.inputs().iter() {
        let input_info = signature_def
            .get_input(input.0)
            .expect("Specified tensor name not found");
        let input_op = graph.operation_by_name_required(&input_info.name().name)?;
        match input_info.dtype() {
            DataType::Int32 => {
                // Swapping num_rows and num_cols to satisfy shape
                let tensor = Tensor::<i32>::new(&[
                    model_input.integer_features.shape.1 as u64,
                    model_input.integer_features.shape.0 as u64,
                ])
                .with_values(&model_input.integer_features.values)?;
                int_tensors.push((input_op, tensor));
            }
            DataType::Float => {
                // Swapping num_rows and num_cols to satisfy shape
                let tensor = Tensor::<f32>::new(&[
                    model_input.float_features.shape.1 as u64,
                    model_input.float_features.shape.0 as u64,
                ])
                .with_values(&model_input.float_features.values)?;
                float_tensors.push((input_op, tensor));
            }
            DataType::String => {
                // Swapping num_rows and num_cols to satisfy shape
                let tensor = Tensor::<String>::new(&[
                    model_input.string_features.shape.1 as u64,
                    model_input.string_features.shape.0 as u64,
                ])
                .with_values(&model_input.string_features.values)?;
                string_tensors.push((input_op, tensor));
            }
            _ => {
                tracing::error!("Type not supported");
                anyhow::bail!("Type not supported")
            }
        }
    }
    Ok(TensorflowModelInput {
        int_tensors,
        float_tensors,
        string_tensors,
    })
}

/// Parses a `ModelInput` into a `TensorflowModelInput` when the TensorFlow model has multiple input signatures.
///
/// # Arguments
/// * `model_inputs` - The `ModelInput` containing the input data.
/// * `signature_def` - The TensorFlow `SignatureDef` that describes the model's input signature.
/// * `graph` - The TensorFlow `Graph` containing the model.
///
/// # Returns
/// * `Ok(TensorflowModelInput)` - If parsing was successful.
/// * `Err(anyhow::Error)` - If there was an error during parsing.
#[tracing::instrument(skip(model_input, signature_def, graph))]
fn parse_functional(
    model_input: ModelInput,
    signature_def: &SignatureDef,
    graph: &Graph,
) -> anyhow::Result<TensorflowModelInput> {
    let mut int_tensors: Vec<(Operation, Tensor<i32>)> = Vec::with_capacity(FEATURE_NAMES_CAPACITY);
    let mut float_tensors: Vec<(Operation, Tensor<f32>)> = Vec::with_capacity(FEATURE_NAMES_CAPACITY * 2);
    let mut string_tensors: Vec<(Operation, Tensor<String>)> = Vec::with_capacity(FEATURE_NAMES_CAPACITY);

    for input in signature_def.inputs().iter() {
        let input_info = match signature_def.get_input(input.0) {
            Ok(input_info) => input_info,
            Err(_) => {
                tracing::error!("Specified tensor with name {} not found", input.0);
                anyhow::bail!("Specified tensor with name {} not found", input.0)
            }
        };
        let input_name = &input_info.name().name;
        let input_op = match graph.operation_by_name_required(input_name) {
            Ok(op) => op,
            Err(_) => {
                tracing::error!("Failed to get input operation {}", input_name);
                anyhow::bail!("Failed to get input operation {}", input_name)
            }
        };
        let model_input_feature_name = match input_name
            .strip_prefix(format!("{}_", DEFAULT_SERVING_SIGNATURE_DEF_KEY).as_str())
        {
            None => {
                tracing::error!("Failed to strip prefix");
                anyhow::bail!("Failed to strip prefix")
            }
            Some(v) => v.to_owned(),
        };

        match input_info.dtype() {
            DataType::Int32 => {
                match get_integer_feature_from_model_input(&model_input, &model_input_feature_name)
                {
                    None => {
                        tracing::error!("Failed to retrieve {} values", model_input_feature_name);
                        anyhow::bail!("Failed to retrieve {} values", model_input_feature_name)
                    }
                    Some(values) => {
                        match Tensor::<i32>::new(&[values.len() as u64, 1]).with_values(values) {
                            Ok(tensor) => {
                                int_tensors.push((input_op, tensor));
                            }
                            Err(_) => {
                                tracing::error!("Failed to populate tensor with int32 values");
                                anyhow::bail!("Failed to populate tensor with int32 values")
                            }
                        };
                    }
                }
            }
            DataType::Float => {
                match get_float_feature_from_model_input(&model_input, &model_input_feature_name) {
                    None => {
                        tracing::error!("Failed to retrieve {} values", model_input_feature_name);
                        anyhow::bail!("Failed to retrieve {} values", model_input_feature_name)
                    }
                    Some(values) => {
                        match Tensor::<f32>::new(&[values.len() as u64, 1]).with_values(values) {
                            Ok(tensor) => {
                                float_tensors.push((input_op, tensor));
                            }
                            Err(_) => {
                                tracing::error!("Failed to populate tensor with float32 values");
                                anyhow::bail!("Failed to populate tensor with float32 values")
                            }
                        };
                    }
                }
            }
            DataType::String => {
                match get_string_feature_from_model_input(&model_input, &model_input_feature_name) {
                    None => {
                        tracing::error!("Failed to retrieve {} values", model_input_feature_name);
                        anyhow::bail!("Failed to retrieve {} values", model_input_feature_name)
                    }
                    Some(values) => {
                        match Tensor::<String>::new(&[values.len() as u64, 1]).with_values(values) {
                            Ok(tensor) => {
                                string_tensors.push((input_op, tensor));
                            }
                            Err(_) => {
                                tracing::error!("Failed to populate tensor with string values");
                                anyhow::bail!("Failed to populate tensor with string values")
                            }
                        };
                    }
                }
            }
            _ => {
                tracing::error!("Type not supported");
                anyhow::bail!("Type not supported")
            }
        }
    }
    Ok(TensorflowModelInput {
        int_tensors,
        float_tensors,
        string_tensors,
    })
}

fn get_integer_feature_from_model_input<'a>(
    model_input: &'a ModelInput,
    feature_name: &FeatureName,
) -> Option<&'a [i32]> {
    let num_features = model_input.integer_features.shape.1;
    if let Some(index) = model_input
        .integer_features
        .names
        .iter()
        .position(|x| x == feature_name)
    {
        let start = if index == 0 { 0 } else { index * num_features };
        let end = start + num_features;
        Some(&model_input.integer_features.values[start..end])
    } else {
        None
    }
}

fn get_float_feature_from_model_input<'a>(
    model_input: &'a ModelInput,
    feature_name: &FeatureName,
) -> Option<&'a [f32]> {
    let num_features = model_input.float_features.shape.1;
    if let Some(index) = model_input
        .float_features
        .names
        .iter()
        .position(|x| x == feature_name)
    {
        let start = if index == 0 { 0 } else { index * num_features };
        let end = start + num_features;
        Some(&model_input.float_features.values[start..end])
    } else {
        None
    }
}

fn get_string_feature_from_model_input<'a>(
    model_input: &'a ModelInput,
    feature_name: &FeatureName,
) -> Option<&'a [String]> {
    let num_features = model_input.string_features.shape.1;
    if let Some(index) = model_input
        .string_features
        .names
        .iter()
        .position(|x| x == feature_name)
    {
        let start = if index == 0 { 0 } else { index * num_features };
        let end = start + num_features;
        Some(&model_input.string_features.values[start..end])
    } else {
        None
    }
}

/// Struct representing a TensorFlow model and its associated components.
pub struct Tensorflow {
    /// TensorFlow graph containing the model structure and operations.
    graph: Graph,
    /// SavedModelBundle containing the loaded model.
    bundle: SavedModelBundle,
    /// SignatureDef describing the model's input and output signatures.
    signature_def: SignatureDef,
    output_names: Vec<String>,
}

impl Tensorflow {
    /// Loads a TensorFlow model from the specified directory.
    ///
    /// # Arguments
    /// * `model_dir` - Directory path where the TensorFlow model is saved.
    ///
    /// # Returns
    /// * `Ok(Tensorflow)` - If the model loading was successful.
    /// * `Err(anyhow::Error)` - If there was an error loading the model.
    #[tracing::instrument]
    pub fn load(model_dir: &str) -> anyhow::Result<Self> {
        const MODEL_TAG: &str = "serve";
        let mut graph = Graph::new();
        let bundle = match SavedModelBundle::load(
            &SessionOptions::new(),
            [MODEL_TAG],
            &mut graph,
            model_dir,
        ) {
            Ok(b) => b,
            Err(_) => {
                tracing::error!("Failed to load TensorFlow model from dir: {}", model_dir);
                anyhow::bail!("Failed to load TensorFlow model from dir: {}", model_dir);
            }
        };

        let signature_def = match bundle
            .meta_graph_def()
            .get_signature(DEFAULT_SERVING_SIGNATURE_DEF_KEY)
        {
            Ok(s) => s.to_owned(),
            Err(_) => {
                tracing::error!(
                    "Failed to get model signature for {}",
                    DEFAULT_SERVING_SIGNATURE_DEF_KEY
                );
                anyhow::bail!(
                    "Failed to get model signature for {}",
                    DEFAULT_SERVING_SIGNATURE_DEF_KEY
                )
            }
        };

        // store output names here as his will avoid allocations on each request
        let mut output_names: Vec<String> = Vec::with_capacity(MAX_OUTPUT_NODES_SUPPORTED);
        for output_def in signature_def.outputs() {
            output_names.push(output_def.0.to_string());
        }

        Ok(Tensorflow {
            graph,
            bundle,
            signature_def,
            output_names,
        })
    }
}

impl Predict for Tensorflow {
    /// Performs prediction using the TensorFlow model.
    ///
    /// # Arguments
    /// * `input` - Input data for prediction, encapsulated in a `ModelInput`.
    ///
    /// # Returns
    /// * `Ok(Output)` - If prediction was successful, containing the predicted output.
    /// * `Err(anyhow::Error)` - If there was an error during prediction.
    #[tracing::instrument(skip(self, input))]
    fn predict(&self, input: ModelInput) -> anyhow::Result<Output> {
        // Parse input into TensorFlow model input format
        let input = TensorflowModelInput::parse(input, &self.signature_def, &self.graph)?;

        // Create session run arguments
        let mut run_args = SessionRunArgs::new();

        // Add input tensors to the session run arguments
        for float_feature in input.float_tensors.iter() {
            run_args.add_feed(&float_feature.0, 0, &float_feature.1);
        }

        for int_feature in input.int_tensors.iter() {
            run_args.add_feed(&int_feature.0, 0, &int_feature.1);
        }

        for string_feature in input.string_tensors.iter() {
            run_args.add_feed(&string_feature.0, 0, &string_feature.1);
        }

        // Prepare output tensors
        let mut fetch_tokens: Vec<FetchToken> = Vec::with_capacity(MAX_OUTPUT_NODES_SUPPORTED);
        for output_def in self.signature_def.outputs() {
            let output_operation = self
                .graph
                .operation_by_name_required(&output_def.1.name().name)?;
            let fetch_token = run_args.request_fetch(&output_operation, output_def.1.name().index);
            fetch_tokens.push(fetch_token);
        }

        // Execute the TensorFlow graph
        match self.bundle.session.run(&mut run_args) {
            Ok(_) => {}
            Err(_) => {
                tracing::error!("Failed to execute TensorFlow graph");
                anyhow::bail!("Failed to execute TensorFlow graph")
            }
        };

        // Retrieve and process the output tensors
        let mut predictions: HashMap<String, Vec<Vec<f64>>> = HashMap::new();

        for (i, token) in fetch_tokens.into_iter().enumerate() {
            let output: Tensor<f32> = run_args.fetch(token)?;

            // model output can have a scaler or nd-array output. Currently, only 2D is supported
            if output.dims().len() > 2 {
                anyhow::bail!("Only 2D shapes are supported in output nodes !")
            }

            // handle non scaler output - is_empty() is true for scalar values
            if !output.dims().is_empty() {
                let processed_output: Vec<Vec<f32>> = output
                    .chunks(output.dims()[1] as usize)
                    .map(|row| row.to_vec())
                    .collect();

                // Convert processed output to the expected format
                let values: Vec<Vec<f64>> = processed_output
                    .iter()
                    .map(|row| {
                        row.iter()
                            .map(|&value| {
                                if value.is_nan() {
                                    -999.99f64
                                } else {
                                    value as f64
                                }
                            })
                            .collect()
                    })
                    .collect();

                predictions.insert(self.output_names.get(i).unwrap().to_string(), values);
            } else {
                // get the scaler value and convert it to Vec<Vec<f64>>
                let scalar_value_vec = output.to_vec();
                let scalar_value = match scalar_value_vec.first() {
                    None => {
                        tracing::error!("Failed to fetch scaler value from output");
                        anyhow::bail!("Failed to fetch scaler value from output")
                    }
                    Some(scaler_value) => scaler_value,
                };
                predictions.insert(
                    self.output_names.get(i).unwrap().to_string(),
                    vec![vec![*scalar_value as f64]],
                );
            }
        }

        Ok(Output { predictions })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::test_utils;

    #[test]
    fn fails_to_load_tensorflow_model() {
        let model_dir = "incorrect/path";
        let model = Tensorflow::load(model_dir);

        // assert the result is Ok
        assert!(model.is_err())
    }

    #[test]
    fn successfully_load_tensorflow_regression_model() {
        let model_dir = "tests/model_storage/models/tensorflow-my_awesome_autompg_model";
        let model = Tensorflow::load(model_dir);

        // assert the result is Ok
        assert!(model.is_ok())
    }

    #[test]
    fn successfully_make_prediction_using_tensorflow_regression_model_when_input_is_tabular_data() {
        let model_dir = "tests/model_storage/models/tensorflow-my_awesome_autompg_model";
        let model = Tensorflow::load(model_dir).unwrap();

        let size = 10;
        let model_inputs = test_utils::utils::create_model_inputs(9, 0, size);

        // make predictions
        let output = model.predict(model_inputs);

        // assert
        assert!(output.is_ok());
        let predictions = output.unwrap().predictions;
        let predictions = predictions.get("dense_2").unwrap(); // the caller should be aware of which key to use

        // asserts the output length of predictions is equal to input length
        assert_eq!(predictions.len(), size);
        // since the predictions is of type Vec<Vec<f64>>, we will assert that inner vec is of
        // length 1 i.e [[1], [2], [3]]. This is because this is a regression model with single output
        assert_eq!(predictions.first().unwrap().len(), 1);
    }

    #[test]
    fn successfully_load_tensorflow_multi_classification_model() {
        let model_dir = "tests/model_storage/models/tensorflow-my_awesome_sequential_model";
        let model = Tensorflow::load(model_dir);

        // assert the result is Ok
        assert!(model.is_ok())
    }

    #[test]
    fn successfully_make_prediction_using_tensorflow_multi_class_classification_model_when_input_is_tabular_data(
    ) {
        let model_dir = "tests/model_storage/models/tensorflow-my_awesome_sequential_model";
        let model = Tensorflow::load(model_dir).unwrap();

        let size = 10;
        let model_inputs = test_utils::utils::create_model_inputs(6, 0, size);

        // make predictions
        let output = model.predict(model_inputs);

        // assert
        assert!(output.is_ok());
        let predictions = output.unwrap().predictions;
        let predictions = predictions.get("output_0").unwrap(); // the caller should be aware of which key to use

        // asserts the output length of predictions is equal to input length
        assert_eq!(predictions.len(), size);
        // since the predictions is of type Vec<Vec<f64>>, we will assert that inner vec is of
        // length 3 i.e [[1,2,3], [1,2,3], [1,2,3]]. This is because this is multi class classification
        // model with 3 classes
        assert_eq!(predictions.first().unwrap().len(), 3);
    }

    #[test]
    fn successfully_load_tensorflow_multi_classification_functional_model_with_multiple_inputs_and_input_is_tabular_data(
    ) {
        let model_dir = "tests/model_storage/models/tensorflow-my_awesome_penguin_model";
        let model = Tensorflow::load(model_dir).unwrap();

        let numeric_feature_names = vec![
            "flipper_length_mm".to_string(),
            "body_mass_g".to_string(),
            "bill_length_mm".to_string(),
            "bill_depth_mm".to_string(),
            "sex".to_string(),
            "island".to_string(),
        ];

        let size = 10;
        let model_inputs =
            test_utils::utils::create_model_inputs_with_names(numeric_feature_names, vec![], size);

        // make predictions
        let output = model.predict(model_inputs);

        // assert
        assert!(output.is_ok());
        let predictions = output.unwrap().predictions;
        let predictions = predictions.get("species").unwrap(); // the caller should be aware of which key to use

        // asserts the output length of predictions is equal to input length
        assert_eq!(predictions.len(), size);
        // since the predictions is of type Vec<Vec<f64>>, we will assert that inner vec is of
        // length 3 i.e [[1,2,3], [1,2,3], [1,2,3]]. This is because this is multi class classification
        // model with 3 classes
        assert_eq!(predictions.first().unwrap().len(), 3);
    }
}
