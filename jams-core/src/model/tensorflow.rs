use crate::model::predictor::{ModelInput, Output, Predictor, Value, Values};
use tensorflow::{
    DataType, Graph, Operation, SavedModelBundle, SessionOptions, SessionRunArgs, SignatureDef,
    Tensor, DEFAULT_SERVING_SIGNATURE_DEF_KEY,
};

/// Struct representing the input tensors for a TensorFlow model.
///
/// This struct encapsulates tensors grouped by their data type (`i32`, `f32`, `String`),
/// along with corresponding TensorFlow operations.
pub struct TensorflowModelInput {
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
    pub fn parse(
        input: ModelInput,
        signature_def: &SignatureDef,
        graph: &Graph,
    ) -> anyhow::Result<Self> {
        let signature_def_num_values = signature_def.inputs().len();
        if signature_def_num_values == 0 {
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
/// * `input` - The `ModelInput` containing the input data.
/// * `signature_def` - The TensorFlow `SignatureDef` that describes the model's input signature.
/// * `graph` - The TensorFlow `Graph` containing the model.
///
/// # Returns
/// * `Ok(TensorflowModelInput)` - If parsing was successful.
/// * `Err(anyhow::Error)` - If there was an error during parsing.
fn parse_sequential(
    input: ModelInput,
    signature_def: &SignatureDef,
    graph: &Graph,
) -> anyhow::Result<TensorflowModelInput> {
    let mut int_features: Vec<Vec<i32>> = Vec::new();
    let mut float_features: Vec<Vec<f32>> = Vec::new();
    let mut string_features: Vec<Vec<String>> = Vec::new();

    // Extract the values from hashmap
    let input_matrix: Vec<Values> = input.values();

    for values in input_matrix {
        // Get the value type
        let first = values.0.first().unwrap();

        // Strings values are pushed to separate vector of type Vec<String>
        // Int and float are pushed to separate of type Vec<f32>
        match first {
            Value::String(_) => {
                string_features.push(values.to_strings());
            }
            Value::Int(_) => {
                int_features.push(values.to_ints());
            }
            Value::Float(_) => {
                float_features.push(values.to_floats());
            }
        }
    }

    // Calculate dimensions of the 2D vector
    let (int_num_rows, int_num_cols) = get_shape(&int_features);
    let (float_num_rows, float_num_cols) = get_shape(&float_features);
    let (string_num_rows, string_num_cols) = get_shape(&string_features);

    // Flatten features
    let flatten_float: Vec<f32> = float_features.into_iter().flatten().collect();
    let flatten_int: Vec<i32> = int_features.into_iter().flatten().collect();
    let flatten_string: Vec<String> = string_features.into_iter().flatten().collect();

    let mut int_tensors: Vec<(Operation, Tensor<i32>)> = Vec::new();
    let mut float_tensors: Vec<(Operation, Tensor<f32>)> = Vec::new();
    let mut string_tensors: Vec<(Operation, Tensor<String>)> = Vec::new();

    // Create tensors
    for input in signature_def.inputs().iter() {
        let input_info = signature_def
            .get_input(input.0)
            .expect("Specified tensor name not found");
        let input_op = graph
            .operation_by_name_required(&input_info.name().name)
            .unwrap();
        match input_info.dtype() {
            DataType::Int32 => {
                // Swapping num_rows and num_cols to satisfy shape
                let tensor = Tensor::<i32>::new(&[int_num_cols as u64, int_num_rows as u64])
                    .with_values(&flatten_int)?;
                int_tensors.push((input_op, tensor));
            }
            DataType::Float => {
                // Swapping num_rows and num_cols to satisfy shape
                let tensor = Tensor::<f32>::new(&[float_num_cols as u64, float_num_rows as u64])
                    .with_values(&flatten_float)?;
                float_tensors.push((input_op, tensor));
            }
            DataType::String => {
                // Swapping num_rows and num_cols to satisfy shape
                let tensor =
                    Tensor::<String>::new(&[string_num_cols as u64, string_num_rows as u64])
                        .with_values(&flatten_string)?;
                string_tensors.push((input_op, tensor));
            }
            _ => {
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
fn parse_functional(
    model_inputs: ModelInput,
    signature_def: &SignatureDef,
    graph: &Graph,
) -> anyhow::Result<TensorflowModelInput> {
    let mut int_tensors: Vec<(Operation, Tensor<i32>)> = Vec::new();
    let mut float_tensors: Vec<(Operation, Tensor<f32>)> = Vec::new();
    let mut string_tensors: Vec<(Operation, Tensor<String>)> = Vec::new();

    for input in signature_def.inputs().iter() {
        let input_info = signature_def
            .get_input(input.0)
            .expect("Specified tensor name not found");
        let input_name = &input_info.name().name;
        let model_input_feature_name = input_name
            .strip_prefix(format!("{}_", DEFAULT_SERVING_SIGNATURE_DEF_KEY).as_str())
            .unwrap()
            .to_owned();

        match input_info.dtype() {
            DataType::Int32 => {
                let values = model_inputs
                    .get(&model_input_feature_name)
                    .unwrap()
                    .to_owned();
                let values_length = values.iter().len() as u64;
                let input_op = graph.operation_by_name_required(input_name).unwrap();

                let tensor = Tensor::<i32>::new(&[values_length, 1])
                    .with_values(&values.to_ints())
                    .unwrap();
                int_tensors.push((input_op, tensor));
            }
            DataType::Float => {
                let values = model_inputs
                    .get(&model_input_feature_name)
                    .unwrap()
                    .to_owned();
                let values_length = values.iter().len() as u64;
                let input_op = graph.operation_by_name_required(input_name).unwrap();

                let tensor = Tensor::<f32>::new(&[values_length, 1])
                    .with_values(&values.to_floats())
                    .unwrap();
                float_tensors.push((input_op, tensor));
            }
            DataType::String => {
                let values = model_inputs
                    .get(&model_input_feature_name)
                    .unwrap()
                    .to_owned();
                let values_length = values.iter().len() as u64;
                let input_op = graph.operation_by_name_required(input_name).unwrap();

                let tensor = Tensor::<String>::new(&[values_length, 1])
                    .with_values(&values.to_strings())
                    .unwrap();
                string_tensors.push((input_op, tensor));
            }
            _ => {
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

/// Returns the shape (rows, cols) of a 2D vector.
///
/// # Arguments
///
/// * `vector` - Reference to a vector of vectors.
///
/// # Returns
///
/// A tuple representing the number of rows and columns in the input vector.
fn get_shape<T>(input: &[Vec<T>]) -> (usize, usize) {
    if !input.is_empty() {
        return (input.len(), input[0].len());
    }
    (0, 0)
}

/// Struct representing a TensorFlow model and its associated components.
pub struct Tensorflow {
    /// TensorFlow graph containing the model structure and operations.
    graph: Graph,
    /// SavedModelBundle containing the loaded model.
    bundle: SavedModelBundle,
    /// SignatureDef describing the model's input and output signatures.
    signature_def: SignatureDef,
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
    pub fn load(model_dir: &str) -> anyhow::Result<Self> {
        const MODEL_TAG: &str = "serve";
        let mut graph = Graph::new();
        let bundle =
            SavedModelBundle::load(&SessionOptions::new(), [MODEL_TAG], &mut graph, model_dir)
                .expect("Failed to load TensorFlow model");
        let signature_def = bundle
            .meta_graph_def()
            .get_signature(DEFAULT_SERVING_SIGNATURE_DEF_KEY)
            .expect("Failed to get model signature")
            .to_owned();
        Ok(Tensorflow {
            graph,
            bundle,
            signature_def,
        })
    }
}

impl Predictor for Tensorflow {
    /// Performs prediction using the TensorFlow model.
    ///
    /// # Arguments
    /// * `input` - Input data for prediction, encapsulated in a `ModelInput`.
    ///
    /// # Returns
    /// * `Ok(Output)` - If prediction was successful, containing the predicted output.
    /// * `Err(anyhow::Error)` - If there was an error during prediction.
    fn predict(&self, input: ModelInput) -> anyhow::Result<Output> {
        // Parse input into TensorFlow model input format
        let input = TensorflowModelInput::parse(input, &self.signature_def, &self.graph)?;

        // Create separate vectors for tensor values
        let float_tensors: Vec<Tensor<f32>> =
            input.float_tensors.iter().map(|(_, t)| t.clone()).collect();
        let int_tensors: Vec<Tensor<i32>> =
            input.int_tensors.iter().map(|(_, t)| t.clone()).collect();
        let string_tensors: Vec<Tensor<String>> = input
            .string_tensors
            .iter()
            .map(|(_, t)| t.clone())
            .collect();

        // Create session run arguments
        let mut run_args = SessionRunArgs::new();

        // Add input tensors to the session run arguments
        for (i, float_feature) in input.float_tensors.iter().enumerate() {
            run_args.add_feed(&float_feature.0, 0, &float_tensors[i]);
        }

        for (i, int_feature) in input.int_tensors.iter().enumerate() {
            run_args.add_feed(&int_feature.0, 0, &int_tensors[i]);
        }

        for (i, string_feature) in input.string_tensors.iter().enumerate() {
            run_args.add_feed(&string_feature.0, 0, &string_tensors[i]);
        }

        // Prepare output tensor
        let outputs_tensor_names: Vec<String> = self
            .signature_def
            .outputs()
            .values()
            .map(|t| t.name().name.to_string())
            .collect();
        let name = outputs_tensor_names.first().unwrap();
        let output_op = self.graph.operation_by_name_required(name).unwrap();
        let output_fetch = run_args.request_fetch(&output_op, 0);

        // Execute the TensorFlow graph
        self.bundle
            .session
            .run(&mut run_args)
            .expect("Failed to execute TensorFlow graph");

        // Retrieve and process the output tensor
        let output: Tensor<f32> = run_args
            .fetch(output_fetch)
            .expect("Failed to fetch output tensor");
        let processed_output: Vec<Vec<f32>> = output
            .chunks(output.dims()[1] as usize)
            .map(|row| row.to_vec())
            .collect();

        // Convert processed output to the expected format
        let predictions: Vec<Vec<f64>> = processed_output
            .iter()
            .map(|row| row.iter().map(|&value| value as f64).collect())
            .collect();

        Ok(Output { predictions })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::test_utils;

    #[test]
    fn successfully_load_tensorflow_regression_model() {
        let model_dir = "tests/model_storage/local_model_store/tensorflow-my_awesome_autompg_model";
        let model = Tensorflow::load(model_dir);

        // assert the result is Ok
        assert!(model.is_ok())
    }

    #[test]
    fn successfully_make_prediction_using_tensorflow_regression_model_when_input_is_tabular_data() {
        let model_dir = "tests/model_storage/local_model_store/tensorflow-my_awesome_autompg_model";
        let model = Tensorflow::load(model_dir).unwrap();

        let size = 10;
        let model_inputs = test_utils::utils::create_model_inputs(9, 0, size);

        // make predictions
        let output = model.predict(model_inputs);

        // assert
        assert!(output.is_ok());
        let predictions = output.unwrap().predictions;

        // asserts the output length of predictions is equal to input length
        assert_eq!(predictions.len(), size);
        // since the predictions is of type Vec<Vec<f64>>, we will assert that inner vec is of
        // length 1 i.e [[1], [2], [3]]. This is because this is a regression model with single output
        assert_eq!(predictions.first().unwrap().len(), 1);
    }

    #[test]
    fn successfully_load_tensorflow_multi_classification_model() {
        let model_dir =
            "tests/model_storage/local_model_store/tensorflow-my_awesome_sequential_model";
        let model = Tensorflow::load(model_dir);

        // assert the result is Ok
        assert!(model.is_ok())
    }

    #[test]
    fn successfully_make_prediction_using_tensorflow_multi_class_classification_model_when_input_is_tabular_data(
    ) {
        let model_dir =
            "tests/model_storage/local_model_store/tensorflow-my_awesome_sequential_model";
        let model = Tensorflow::load(model_dir).unwrap();

        let size = 10;
        let model_inputs = test_utils::utils::create_model_inputs(6, 0, size);

        // make predictions
        let output = model.predict(model_inputs);

        // assert
        assert!(output.is_ok());
        let predictions = output.unwrap().predictions;

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
        let model_dir = "tests/model_storage/local_model_store/tensorflow-my_awesome_penguin_model";
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

        // asserts the output length of predictions is equal to input length
        assert_eq!(predictions.len(), size);
        // since the predictions is of type Vec<Vec<f64>>, we will assert that inner vec is of
        // length 3 i.e [[1,2,3], [1,2,3], [1,2,3]]. This is because this is multi class classification
        // model with 3 classes
        assert_eq!(predictions.first().unwrap().len(), 3);
    }
}
