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
