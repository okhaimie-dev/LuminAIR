use std::{
    any::{Any, TypeId},
    sync::Arc,
};

use luminal::op::InputTensor;

use crate::data::StwoData;

/// Checks if a `TypeId` matches the type `T`.
pub(crate) fn is<T: Any>(type_id: TypeId) -> bool {
    type_id == TypeId::of::<T>()
}

pub(crate) fn get_data(tensor: &InputTensor) -> StwoData {
    // With the copy operators in place, inputs should already be StwoData
    if let Some(data) = tensor.borrowed().downcast_ref::<StwoData>() {
        // Most common case with CopyToStwo in place - just clone the reference
        StwoData(Arc::clone(&data.0))
    } else if let Some(data) = tensor.borrowed().downcast_ref::<Vec<f32>>() {
        // Fallback for direct Vec<f32> inputs (should be rare)
        StwoData::from_f32(data)
    } else {
        panic!("Unsupported input type: expected StwoData or Vec<f32>");
    }
}
