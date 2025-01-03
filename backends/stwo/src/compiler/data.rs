use std::any::Any;

use luminal::prelude::*;
use stwo_prover::core::{backend::simd::m31::PackedBaseField, fields::m31::BaseField};

use crate::air::tensor::TensorField;

// Base trait that defines what a tensor's data structure must provide
// It combines the ability to store and access field elements.
pub trait TensorData: Data {
    // Associated type specifying which field type this data container uses
    // Will be either PackedBaseField (for SIMD) or BaseField (for CPU)
    type Field: TensorField;

    // Method to access the underlying data
    fn as_slice(&self) -> &[Self::Field];
}

// Trait for converting from a vector of field elements back into a tensor data structure
pub trait FromTensorField<F: TensorField> {
    fn from_tensor_field(data: Vec<F>) -> Self;
}

////////////// SIMD BACKEND //////////////

// Data structure for SIMD operations
#[derive(Debug, Clone)]
pub struct PackedData(pub Vec<PackedBaseField>);

// Required implementation to make PackedData usable with luminal's type system
impl Data for PackedData {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// Implements the tensor data interface for PackedData
impl TensorData for PackedData {
    type Field = PackedBaseField;
    fn as_slice(&self) -> &[Self::Field] {
        &self.0
    }
}

// Implementation for SIMD operations - converts packed field vector to PackedData
impl FromTensorField<PackedBaseField> for PackedData {
    fn from_tensor_field(data: Vec<PackedBaseField>) -> Self {
        PackedData(data) // Simply wrap the vector
    }
}

////////////// CPU BACKEND //////////////

// Data structure for CPU operations
#[derive(Debug, Clone)]
pub struct NonPackedData(pub Vec<BaseField>);

// Required implementation to make NonPackedData usable with luminal's type system
impl Data for NonPackedData {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// Implements tensor data interface for CPU operations
impl TensorData for NonPackedData {
    type Field = BaseField;
    fn as_slice(&self) -> &[Self::Field] {
        &self.0
    }
}

// Implementation for CPU operations - converts regular field vector to NonPackedData
impl FromTensorField<BaseField> for NonPackedData {
    fn from_tensor_field(data: Vec<BaseField>) -> Self {
        NonPackedData(data) // Simply wrap the vector
    }
}
