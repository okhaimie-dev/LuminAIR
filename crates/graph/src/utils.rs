use std::any::{Any, TypeId};

use luminal::prelude::*;

use crate::data::StwoData;

/// Checks if a `TypeId` matches the type `T`.
pub(crate) fn is<T: Any>(type_id: TypeId) -> bool {
    type_id == TypeId::of::<T>()
}

/// Extracts the StwoData reference from an InputTensor.
pub(crate) fn get_buffer_from_tensor<'a>(tensor: &'a InputTensor) -> &'a StwoData {
    &tensor.borrowed().downcast_ref::<StwoData>().unwrap()
}
