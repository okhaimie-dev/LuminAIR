use luminal::{op::InputTensor, shape::Expression};

/// Helper function to extract Vec<f32> from InputTensor
pub(crate) fn get_vec<'a>(tensor: &'a InputTensor<'a>) -> &'a Vec<f32> {
    tensor
        .borrowed()
        .downcast_ref::<Vec<f32>>()
        .expect("Tensor data is not Vec<f32>")
}

pub(crate) fn get_index(
    data: &[f32],
    (ind, val): &(Expression, Expression),
    stack: &mut Vec<i64>,
    index: usize,
) -> f32 {
    if val.exec_single_var_stack(index, stack) != 0 {
        data[ind.exec_single_var_stack(index, stack)]
    } else {
        0.0
    }
}
