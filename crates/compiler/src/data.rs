use luminal::prelude::*;
use std::{fmt::Debug, sync::Arc};
use stwo_prover::core::backend::simd::m31::PackedBaseField;

use crate::utils::{pack_floats, unpack_floats};

#[derive(Clone, Debug)]
pub struct StwoData(pub Arc<Vec<PackedBaseField>>);

impl StwoData {
    pub fn as_slice(&self) -> &[PackedBaseField] {
        &self.0
    }

    pub fn from_f32(data: &[f32]) -> Self {
        let packed = pack_floats(data);
        StwoData(Arc::new(packed))
    }

    pub fn to_f32(&self, len: usize) -> Vec<f32> {
        unpack_floats(&self.0, len)
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
        Self { data, output_size }
    }

    pub fn to_f32(&self) -> Vec<f32> {
        // Convert only the final output from fixed point to f32
        unpack_floats(&self.data.0, self.output_size)
    }
}

// Trait for converting graph output
pub trait GraphOutputConverter {
    fn get_final_output(&mut self, id: NodeIndex) -> Vec<f32>;
}

impl GraphOutputConverter for Graph {
    fn get_final_output(&mut self, id: NodeIndex) -> Vec<f32> {
        // Get the shape from the graph edges
        let output_size = if let Some((_, shape)) = self.to_retrieve.get(&id) {
            shape
                .n_elements()
                .to_usize()
                .expect("Failed to get tensor size")
        } else {
            // Fallback to checking graph edges if not in to_retrieve
            self.graph
                .edges_directed(id, petgraph::Direction::Incoming)
                .find_map(|e| e.weight().as_data())
                .map(|(_, _, shape)| {
                    shape
                        .n_elements()
                        .to_usize()
                        .expect("Failed to get tensor size")
                })
                .expect("Could not determine tensor shape")
        };

        if let Some(tensor) = self.tensors.remove(&(id, 0)) {
            if let Some(data) = tensor.downcast_ref::<StwoData>() {
                let converter = OutputConverter::new(data.clone(), output_size);
                return converter.to_f32();
            }
        }
        panic!("No StwoData found for final output conversion");
    }
}
