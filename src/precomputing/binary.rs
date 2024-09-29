use luminal::{op::InputTensor, shape::ShapeTracker};

use super::helpers::{compute_strides, determine_broadcast_shape, expand_data, get_vec};

pub(crate) fn precompile_binary_op(
    tensors: Vec<(InputTensor, ShapeTracker)>,
) -> (Vec<f32>, Vec<f32>) {
    // Extract input tensors and their ShapeTrackers
    let (tensor_a, shape_a_tracker) = &tensors[0];
    let (tensor_b, shape_b_tracker) = &tensors[1];

    // Downcast to Vec<f32>
    let data_a = get_vec(tensor_a);
    let data_b = get_vec(tensor_b);

    // Get original shapes
    let shape_a = shape_a_tracker.shape_usize();
    let shape_b = shape_b_tracker.shape_usize();

    // Determine broadcasted shape
    let broadcast_shape = match determine_broadcast_shape(&shape_a, &shape_b) {
        Ok(shape) => shape,
        Err(e) => panic!("Broadcasting error: {}", e),
    };

    // Compute strides for original tensors
    let strides_a = compute_strides(&shape_a);
    let strides_b = compute_strides(&shape_b);

    // Expand data according to broadcasted shape
    let expanded_a = expand_data(&data_a, &shape_a, &broadcast_shape, &strides_a);
    let expanded_b = expand_data(&data_b, &shape_b, &broadcast_shape, &strides_b);

    (expanded_a, expanded_b)
}
