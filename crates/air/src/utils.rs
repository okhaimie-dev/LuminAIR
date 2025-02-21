use num_traits::Zero;
use stwo_prover::core::backend::simd::{m31::LOG_N_LANES, qm31::PackedSecureField};

use crate::LuminairInteractionClaim;

pub fn calculate_log_size(max_size: usize) -> u32 {
    ((max_size + (1 << LOG_N_LANES) - 1) >> LOG_N_LANES)
        .next_power_of_two()
        .trailing_zeros()
        + LOG_N_LANES
}

/// Verify that the claims (i.e Statement) are valid.
pub fn lookup_sum_valid(interaction_claim: &LuminairInteractionClaim) -> bool {
    let mut sum = PackedSecureField::zero();

    interaction_claim
        .add
        .iter()
        .for_each(|c| sum += c.claimed_sum.into());
    sum.is_zero()
}
