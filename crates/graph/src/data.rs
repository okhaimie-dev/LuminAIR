use luminal::prelude::*;
use numerair::FixedPoint;
use std::sync::Arc;
use stwo_prover::core::backend::simd::m31::PackedBaseField;

#[derive(Clone, Debug)]
pub(crate) struct StwoData(pub(crate) Arc<Vec<PackedBaseField>>);

impl StwoData {
    #[allow(dead_code)]
    pub(crate) fn as_slice(&self) -> &[PackedBaseField] {
        &self.0
    }

    pub(crate) fn from_f32(data: &[f32]) -> Self {
        let packed = pack_floats(data);
        StwoData(Arc::new(packed))
    }

    #[allow(dead_code)]
    pub(crate) fn to_f32(&self, len: usize) -> Vec<f32> {
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
pub(crate) struct OutputConverter {
    data: StwoData,
    output_size: usize,
}

impl OutputConverter {
    pub(crate) fn new(data: StwoData, output_size: usize) -> Self {
        Self { data, output_size }
    }

    pub(crate) fn to_f32(&self) -> Vec<f32> {
        // Convert only the final output from fixed point to f32
        unpack_floats(&self.data.0, self.output_size)
    }
}

// TODO: Optimize float packing strategy.
// Currently, the same value is broadcast to the entire lane.
// To benefit of SIMD, we should populate the lane with distinct values,
// which allows for more efficient utilization of the packing process.
// Note: This requires careful handling of broadcasting rules during tensor operations
// and ensuring the correct identity value is used based on the specific operation being performed.
fn pack_floats(values: &[f32]) -> Vec<PackedBaseField> {
    let mut packed = Vec::new();
    for value in values {
        packed.push(PackedBaseField::from_f64(*value as f64));
    }
    packed
}

fn unpack_floats(packed: &[PackedBaseField], len: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(len);
    for &packed_value in packed.iter() {
        result.push(packed_value.to_f64() as f32);
        if result.len() == len {
            break;
        }
    }
    result
}
