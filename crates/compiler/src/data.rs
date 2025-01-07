use luminal::prelude::*;
use std::{fmt::Debug, sync::Arc};
use stwo_prover::core::backend::simd::m31::PackedBaseField;

use crate::fixed_point::{pack_floats, unpack_floats, DEFAULT_SCALE};

#[derive(Clone, Debug)]
pub struct StwoData(pub Arc<Vec<PackedBaseField>>);

impl StwoData {
    pub fn as_slice(&self) -> &[PackedBaseField] {
        &self.0
    }

    pub fn from_f32(data: &[f32]) -> Self {
        let packed = pack_floats(data, DEFAULT_SCALE);
        StwoData(Arc::new(packed))
    }

    pub fn to_f32(&self, len: usize) -> Vec<f32> {
        unpack_floats(&self.0, DEFAULT_SCALE, len)
    }
}

impl Data for StwoData {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

// Final output conversion handler
pub struct OutputConverter {
    data: StwoData,
    output_size: usize,
}

impl OutputConverter {
    pub fn new(data: StwoData, output_size: usize) -> Self {
        Self { 
            data,
            output_size 
        }
    }

    pub fn to_f32(&self) -> Vec<f32> {
        // Convert only the final output from fixed point to f32
        unpack_floats(&self.data.0, DEFAULT_SCALE, self.output_size)
    }
}

// Trait for converting graph output
pub trait GraphOutputConverter {
    fn get_final_output(&mut self, id: NodeIndex, output_size: usize) -> Vec<f32>;
}

impl GraphOutputConverter for Graph {
    fn get_final_output(&mut self, id: NodeIndex, output_size: usize) -> Vec<f32> {
        if let Some(tensor) = self.tensors.remove(&(id, 0)) {
            if let Some(data) = tensor.downcast_ref::<StwoData>() {
                let converter = OutputConverter::new(data.clone(), output_size);
                return converter.to_f32();
            }
        }
        panic!("No StwoData found for final output conversion");
    }
}