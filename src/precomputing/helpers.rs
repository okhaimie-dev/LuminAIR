use luminal::{op::InputTensor, shape::ShapeTracker};

/// Helper function to determine broadcasted shape
pub(super) fn determine_broadcast_shape(
    shape_a: &[usize],
    shape_b: &[usize],
) -> Result<Vec<usize>, String> {
    let len_a = shape_a.len();
    let len_b = shape_b.len();
    let len = usize::max(len_a, len_b);
    let mut broadcast_shape = Vec::with_capacity(len);

    for i in 0..len {
        let dim_a = if i < len - len_a {
            1
        } else {
            shape_a[i - (len - len_a)]
        };
        let dim_b = if i < len - len_b {
            1
        } else {
            shape_b[i - (len - len_b)]
        };

        if dim_a != dim_b && dim_a != 1 && dim_b != 1 {
            return Err(format!(
                "Shapes {:?} and {:?} are not broadcastable.",
                shape_a, shape_b
            ));
        }

        broadcast_shape.push(usize::max(dim_a, dim_b));
    }

    Ok(broadcast_shape)
}

/// Helper function to compute strides for a given shape
pub(super) fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len()).rev().skip(1) {
        strides[i - 1] = strides[i] * shape[i];
    }
    strides
}

/// Helper function to expand data according to the broadcasted shape
pub(super) fn expand_data(
    data: &[f32],
    original_shape: &[usize],
    broadcast_shape: &[usize],
    original_strides: &[usize],
) -> Vec<f32> {
    let mut expanded_data = Vec::with_capacity(broadcast_shape.iter().product());

    let broadcast_strides = compute_strides(broadcast_shape);

    let len = broadcast_shape.iter().product();
    for idx in 0..len {
        // Compute multi-dimensional index
        let mut remaining = idx;
        let mut md_index = vec![0; broadcast_shape.len()];
        for (i, _) in broadcast_shape.iter().enumerate() {
            md_index[i] = remaining / broadcast_strides[i];
            remaining %= broadcast_strides[i];
        }

        // Map to original tensor's index
        let mut original_idx = 0;
        for (i, &dim) in original_shape.iter().enumerate() {
            let index = if dim == 1 {
                0
            } else {
                md_index[broadcast_shape.len() - original_shape.len() + i]
            };
            original_idx += index * original_strides[i];
        }

        expanded_data.push(data[original_idx]);
    }

    expanded_data
}

/// Helper function to extract Vec<f32> from InputTensor
pub(crate) fn get_vec<'a>(tensor: &'a InputTensor<'a>) -> &'a Vec<f32> {
    tensor
        .borrowed()
        .downcast_ref::<Vec<f32>>()
        .expect("Tensor data is not Vec<f32>")
}

/// Helper function to compute the effective shape by treating fake dimensions as size 1
pub(super) fn get_effective_shape(shape_tracker: &ShapeTracker) -> Vec<usize> {
    shape_tracker
        .shape_usize()
        .iter()
        .zip(shape_tracker.fake.iter())
        .map(|(dim, &is_fake)| if is_fake { 1 } else { *dim })
        .collect()
}
