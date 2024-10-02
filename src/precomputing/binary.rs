use super::helpers::get_vec;
use luminal::{op::InputTensor, shape::ShapeTracker};

pub(crate) fn precompile_binary_op(
    tensors: Vec<(InputTensor, ShapeTracker)>,
) -> (Vec<f32>, Vec<f32>) {
    // Extract input tensors and their ShapeTrackers
    let (tensor_a, shape_a_tracker) = &tensors[0];
    let (tensor_b, shape_b_tracker) = &tensors[1];

    // Downcast to Vec<f32>
    let data_a = get_vec(tensor_a);
    let data_b = get_vec(tensor_b);

    // Determine broadcasted shape
    let mut broadcast_tracker = shape_a_tracker.clone();
    let b_dims = shape_b_tracker.dims();
    for (i, &dim) in b_dims.iter().enumerate() {
        if i >= broadcast_tracker.len() {
            broadcast_tracker.expand(i, dim);
        } else if broadcast_tracker.dims()[i].to_usize().unwrap() < dim.to_usize().unwrap() {
            broadcast_tracker.expand(i, dim);
        }
    }

    // Expand data according to broadcasted shape
    let expanded_a = expand_data_to_broadcast(data_a, shape_a_tracker, &broadcast_tracker);
    let expanded_b = expand_data_to_broadcast(data_b, shape_b_tracker, &broadcast_tracker);

    (expanded_a, expanded_b)
}

fn expand_data_to_broadcast(
    data: &[f32],
    original_shape: &ShapeTracker,
    broadcast_shape: &ShapeTracker,
) -> Vec<f32> {
    let mut expanded_data = vec![0.0; broadcast_shape.n_elements().to_usize().unwrap()];
    let expr = (
        original_shape.index_expression(),
        original_shape.valid_expression(),
    );
    let mut stack = vec![];

    for i in 0..expanded_data.len() {
        expanded_data[i] = super::helpers::get_index(data, &expr, &mut stack, i);
    }

    expanded_data
}
