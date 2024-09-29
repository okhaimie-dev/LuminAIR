use luminal::{op::InputTensor, shape::ShapeTracker};

use super::helpers::{determine_broadcast_shape, expand_data, get_effective_shape, get_vec};

pub(crate) fn precompile_binary_op(
    tensors: Vec<(InputTensor, ShapeTracker)>,
) -> (Vec<f32>, Vec<f32>) {
    // Extract input tensors and their ShapeTrackers
    let (tensor_a, shape_a_tracker) = &tensors[0];
    let (tensor_b, shape_b_tracker) = &tensors[1];

    // Downcast to Vec<f32>
    let data_a = get_vec(tensor_a);
    let data_b = get_vec(tensor_b);

    // Get effective shapes by treating fake dimensions as size 1
    let shape_a = get_effective_shape(shape_a_tracker);
    let shape_b = get_effective_shape(shape_b_tracker);

    // Determine broadcasted shape
    let broadcast_shape = match determine_broadcast_shape(&shape_a, &shape_b) {
        Ok(shape) => shape,
        Err(e) => panic!("Broadcasting error: {}", e),
    };

    // Compute strides for tensors
    let strides_a_expr = shape_a_tracker.strides();
    let strides_a = strides_a_expr
        .iter()
        .map(|expr| expr.to_usize().unwrap())
        .collect::<Vec<_>>();

    let strides_b_expr = shape_b_tracker.strides();
    let strides_b = strides_b_expr
        .iter()
        .map(|expr| expr.to_usize().unwrap())
        .collect::<Vec<_>>();

    // Expand data according to broadcasted shape
    let expanded_a = expand_data(&data_a, &shape_a, &broadcast_shape, &strides_a);
    let expanded_b = expand_data(&data_b, &shape_b, &broadcast_shape, &strides_b);

    (expanded_a, expanded_b)
}
