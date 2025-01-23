use num_traits::identities::Zero;
use stwo_prover::core::backend::simd::m31::{PackedBaseField, LOG_N_LANES};
use stwo_prover::core::backend::Column;
use stwo_prover::core::{
    backend::{simd::SimdBackend, Col},
    fields::m31::BaseField,
    poly::{
        circle::{CanonicCoset, CircleEvaluation},
        BitReversedOrder,
    },
    ColumnVec,
};

use crate::Claim;

/// Type for trace evaluation to be used in Stwo.
pub type TraceEval = ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>;

/// Claim for the Program trace.
pub type AddClaim = Claim;

pub fn gen_add_trace(
    log_size: u32,
    a: &[PackedBaseField],
    b: &[PackedBaseField],
) -> (TraceEval, AddClaim, Vec<PackedBaseField>) {
    // Calculate trace size and initialize columns
    let trace_size = 1 << log_size;
    let mut trace = Vec::with_capacity(3);
    for _ in 0..3 {
        trace.push(Col::<SimdBackend, BaseField>::zeros(trace_size));
    }

    // Calculate actual size needed
    let size = a.len().max(b.len());

    // Prepare output data
    let mut c_data = Vec::with_capacity(size);

    // Fill trace and generate output data
    for i in 0..trace_size {
        if i < size {
            // Get values with broadcasting
            let a_val = a[i % a.len()];
            let b_val = b[i % b.len()];
            let sum = a_val + b_val;

            trace[0].set(i, a_val.to_array()[0]);
            trace[1].set(i, b_val.to_array()[0]);
            trace[2].set(i, sum.to_array()[0]);

            if i < size {
                c_data.push(sum);
            }
        } else {
            // Pad remaining trace with zeros
            trace[0].set(i, BaseField::zero());
            trace[1].set(i, BaseField::zero());
            trace[2].set(i, BaseField::zero());
        }
    }

    // Create domain
    let domain = CanonicCoset::new(log_size).circle_domain();

    (
        trace
            .into_iter()
            .map(|eval| CircleEvaluation::new(domain, eval))
            .collect(),
        AddClaim { log_size },
        c_data,
    )
}

pub fn calculate_log_size(max_size: usize) -> u32 {
    ((max_size + (1 << LOG_N_LANES) - 1) >> LOG_N_LANES)
        .next_power_of_two()
        .trailing_zeros()
        + LOG_N_LANES
}
