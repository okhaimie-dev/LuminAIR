use luminal::prelude::*;
use stwo_prover::core::backend::simd::m31::PackedBaseField;
use std::{fmt::Debug, sync::Arc};

#[derive(Clone, Debug)]
pub struct StwoData(pub Arc<Vec<PackedBaseField>>);

impl StwoData {
    pub fn as_slice(&self) -> &[PackedBaseField] {
        &self.0
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