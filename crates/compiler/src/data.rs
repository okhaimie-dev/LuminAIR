use luminal::prelude::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use stwo_prover::core::{backend::simd::m31::{PackedBaseField, LOG_N_LANES}, fields::m31::BaseField};
use std::{fmt::Debug, sync::Arc};

#[derive(Clone, Debug)]
pub struct StwoData(pub Arc<Vec<PackedBaseField>>);

impl StwoData {
    pub fn as_slice(&self) -> &[PackedBaseField] {
        &self.0
    }

    pub fn from_f32(data: &[f32]) -> Self {
        let n_lanes = 1 << LOG_N_LANES;
        let n_packed = (data.len() + n_lanes - 1) / n_lanes;

        let packed = (0..n_packed)
            .into_par_iter()
            .map(|i| {
                let start = i * n_lanes;
                let mut values = [0u32; 1 << LOG_N_LANES];
                
                // Fill SIMD lanes
                for (j, val) in values.iter_mut().enumerate() {
                    let idx = start + j;
                    *val = if idx < data.len() {
                        data[idx] as u32  // TODO (@raphaelDkn): Implement fixed point strategy
                    } else {
                        0
                    };
                }
                
                // Convert array to PackedBaseField
                PackedBaseField::from_array(values.map(|x| BaseField::from_u32_unchecked(x)))
            })
            .collect::<Vec<_>>();

        StwoData(Arc::new(packed))
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