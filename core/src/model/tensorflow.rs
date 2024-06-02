use crate::model::predictor::{ModelInput, Output, Predictor, Value, Values};
use tensorflow::{
    DataType, Graph, Operation, SavedModelBundle, SessionOptions, SessionRunArgs, SignatureDef,
    Tensor, DEFAULT_SERVING_SIGNATURE_DEF_KEY,
};

struct TensorflowModelInput {
    int_tensors: Vec<(Operation, Tensor<i32>)>,
    float_tensors: Vec<(Operation, Tensor<f32>)>,
    string_tensors: Vec<(Operation, Tensor<String>)>,
}

impl TensorflowModelInput {
    fn parse(
        input: ModelInput,
        signature_def: &SignatureDef,
        graph: &Graph,
    ) -> anyhow::Result<Self> {
        let signature_def_num_values = signature_def.inputs().len();
        if signature_def_num_values == 0 {
            anyhow::bail!("model graph has no inputs")
        } else if signature_def_num_values == 1 {
            parse_sequential(input, signature_def, graph)
        } else {
            parse_functional(input, signature_def, graph)
        }
    }
}

fn parse_sequential(
    input: ModelInput,
    signature_def: &SignatureDef,
    graph: &Graph,
) -> anyhow::Result<TensorflowModelInput> {
    let mut int_features: Vec<Vec<i32>> = Vec::new();
    let mut float_features: Vec<Vec<f32>> = Vec::new();
    let mut string_features: Vec<Vec<String>> = Vec::new();

    // extract the values from hashmap
    let input_matrix: Vec<Values> = input.values();

    for values in input_matrix {
        // get the value type
        let first = values.0.first().unwrap();

        // strings values are pushed to separate vector of type Vec<String>
        // int and float are pushed to separate of type Vec<f32>
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

    // calculate dimensions of the 2D vector
    let (int_num_rows, int_num_cols) = get_shape(&int_features);
    let (float_num_rows, float_num_cols) = get_shape(&float_features);
    let (string_num_rows, string_num_cols) = get_shape(&string_features);

    // flatten features
    let flatten_float: Vec<f32> = float_features.into_iter().flatten().collect();
    let flatten_int: Vec<i32> = int_features.into_iter().flatten().collect();
    let flatten_string: Vec<String> = string_features.into_iter().flatten().collect();

    let mut int_tensors: Vec<(Operation, Tensor<i32>)> = Vec::new();
    let mut float_tensors: Vec<(Operation, Tensor<f32>)> = Vec::new();
    let mut string_tensors: Vec<(Operation, Tensor<String>)> = Vec::new();

    // create tensors
    for input in signature_def.inputs().iter() {
        let input_info = signature_def
            .get_input(input.0)
            .expect("specified tensor name not found");
        let input_op = graph
            .operation_by_name_required(&input_info.name().name)
            .unwrap();
        match input_info.dtype() {
            DataType::Int32 => {
                // swapping num_rows and num_cols to satisfy shape
                let tensor = Tensor::<i32>::new(&[int_num_cols as u64, int_num_rows as u64])
                    .with_values(&flatten_int)?;
                int_tensors.push((input_op, tensor));
            }
            DataType::Float => {
                // swapping num_rows and num_cols to satisfy shape
                let tensor = Tensor::<f32>::new(&[float_num_cols as u64, float_num_rows as u64])
                    .with_values(&flatten_float)?;
                float_tensors.push((input_op, tensor));
            }
            DataType::String => {
                // swapping num_rows and num_cols to satisfy shape
                let tensor =
                    Tensor::<String>::new(&[string_num_cols as u64, string_num_rows as u64])
                        .with_values(&flatten_string)?;
                string_tensors.push((input_op, tensor));
            }
            _ => {
                anyhow::bail!("type not supported")
            }
        }
    }
    Ok(TensorflowModelInput {
        int_tensors,
        float_tensors,
        string_tensors,
    })
}
fn parse_functional(
    model_inputs: ModelInput,
    signature_def: &SignatureDef,
    graph: &Graph,
) -> anyhow::Result<TensorflowModelInput> {
    let mut int_tensors: Vec<(Operation, Tensor<i32>)> = Vec::new();
    let mut float_tensors: Vec<(Operation, Tensor<f32>)> = Vec::new();
    let mut string_tensors: Vec<(Operation, Tensor<String>)> = Vec::new();

    println!("{:?}", signature_def.inputs());

    for input in signature_def.inputs().iter() {
        let input_info = signature_def
            .get_input(input.0)
            .expect("specified tensor name not found");
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
                anyhow::bail!("type not supported")
            }
        }
    }
    Ok(TensorflowModelInput {
        int_tensors,
        float_tensors,
        string_tensors,
    })
}

// get_shape is a helper functions to return shape of 2D vec
fn get_shape<T>(input: &[Vec<T>]) -> (usize, usize) {
    if !input.is_empty() {
        return (input.len(), input[0].len());
    }
    (0, 0)
}

pub struct Tensorflow {
    graph: Graph,
    bundle: SavedModelBundle,
    signature_def: SignatureDef,
}

impl Tensorflow {
    pub fn load(model_dir: &str) -> anyhow::Result<Self> {
        const MODEL_TAG: &str = "serve";
        let mut graph = Graph::new();
        let bundle =
            SavedModelBundle::load(&SessionOptions::new(), [MODEL_TAG], &mut graph, model_dir)
                .expect("failed to load tensorflow model");
        let signature_def = bundle
            .meta_graph_def()
            .get_signature(DEFAULT_SERVING_SIGNATURE_DEF_KEY)
            .expect("failed to get signature")
            .to_owned();
        Ok(Tensorflow {
            graph,
            bundle,
            signature_def,
        })
    }
}

impl Predictor for Tensorflow {
    fn predict(&self, input: ModelInput) -> anyhow::Result<Output> {
        // parse input
        let input = TensorflowModelInput::parse(input, &self.signature_def, &self.graph)?;

        // create new variables
        // todo: check if this can be avoided
        let float_tensors: Vec<Tensor<f32>> =
            input.float_tensors.iter().map(|(_, t)| t.clone()).collect();
        let int_tensors: Vec<Tensor<i32>> =
            input.int_tensors.iter().map(|(_, t)| t.clone()).collect();
        let string_tensors: Vec<Tensor<String>> = input
            .string_tensors
            .iter()
            .map(|(_, t)| t.clone())
            .collect();

        // create a new session
        let mut run_args = SessionRunArgs::new();

        // update tensor graph for prediction
        for (i, float_feature) in input.float_tensors.iter().enumerate() {
            run_args.add_feed(&float_feature.0, 0, &float_tensors[i]);
        }

        for (i, int_feature) in input.int_tensors.iter().enumerate() {
            run_args.add_feed(&int_feature.0, 0, &int_tensors[i]);
        }

        for (i, string_feature) in input.string_tensors.iter().enumerate() {
            run_args.add_feed(&string_feature.0, 0, &string_tensors[i]);
        }

        // prepare output tensor
        // the len should be at least 1
        // todo: update to support multiple outputs
        let outputs_tensor_names: Vec<String> = self
            .signature_def
            .outputs()
            .values()
            .map(|t| t.name().name.to_string())
            .collect();
        let name = outputs_tensor_names.first().unwrap();
        let output_op = self.graph.operation_by_name_required(name).unwrap();
        let output_fetch = run_args.request_fetch(&output_op, 0);

        // run the graph
        self.bundle
            .session
            .run(&mut run_args)
            .expect("failed to execute tensor graph");

        // retrieve and process the output
        let output: Tensor<f32> = run_args
            .fetch(output_fetch)
            .expect("failed to fetch output");
        let processed_output: Vec<Vec<f32>> = output
            .chunks(output.dims()[1] as usize)
            .map(|row| row.to_vec())
            .collect();

        // convert to Output
        let predictions: Vec<Vec<f64>> = processed_output
            .iter() // Iterate over each row (Vec<f32>)
            .map(|row| {
                row.iter() // Iterate over each element (f32) in the row
                    .map(|&value| value as f64) // Convert f32 to f64
                    .collect() // Collect into Vec<f64>
            })
            .collect(); // Collect into Vec<Vec<f64>>;
        Ok(Output { predictions })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::test_utils;

    #[test]
    fn successfully_load_tensorflow_regression_model() {
        let model_dir = "tests/model_artefacts/autompg_tensorflow";
        let model = Tensorflow::load(model_dir);

        // assert the result is Ok
        assert!(model.is_ok())
    }

    #[test]
    fn successfully_make_prediction_using_tensorflow_regression_model_when_input_is_tabular_data() {
        let model_dir = "tests/model_artefacts/autompg_tensorflow";
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
        let model_dir = "tests/model_artefacts/penguin_tensorflow";
        let model = Tensorflow::load(model_dir);

        // assert the result is Ok
        assert!(model.is_ok())
    }

    #[test]
    fn successfully_make_prediction_using_tensorflow_multi_class_classification_model_when_input_is_tabular_data(
    ) {
        let model_dir = "tests/model_artefacts/penguin_tensorflow";
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
        let model_dir = "tests/model_artefacts/penguin_tensorflow_functional";
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
