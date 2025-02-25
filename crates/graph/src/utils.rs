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
    if let Some(data) = tensor.borrowed().downcast_ref::<Vec<f32>>() {
        StwoData::from_f32(data)
    } else if let Some(data) = tensor.borrowed().downcast_ref::<StwoData>() {
        StwoData(Arc::clone(&data.0))
    } else {
        panic!("Unsupported input type for Add");
    }
}
