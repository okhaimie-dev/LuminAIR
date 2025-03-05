use luminal::shape::Expression;
use num_traits::Zero;
use numerair::packed::FixedPackedBaseField;
use stwo_prover::core::backend::simd::{m31::LOG_N_LANES, qm31::PackedSecureField};

use crate::LuminairInteractionClaim;

/// Calculates the logarithmic size of the trace based on the maximum size of the data.
pub fn calculate_log_size(max_size: usize) -> u32 {
    ((max_size + (1 << LOG_N_LANES) - 1) >> LOG_N_LANES)
        .next_power_of_two()
        .trailing_zeros()
        + LOG_N_LANES
}

/// Verifies the validity of the interaction claim by checking if the sum of claimed sums is zero.
pub fn lookup_sum_valid(interaction_claim: &LuminairInteractionClaim) -> bool {
    let mut sum = PackedSecureField::zero();

    interaction_claim
        .add
        .iter()
        .for_each(|c| sum += c.claimed_sum.into());

    interaction_claim
        .mul
        .iter()
        .for_each(|c| sum += c.claimed_sum.into());

    sum.is_zero()
}

/// Generates a vector of logarithmic sizes for the 'is_first' trace columns.
pub fn get_is_first_log_sizes(max_log_size: u32) -> Vec<u32> {
    let padded_max = max_log_size + 2;
    (4..=padded_max).rev().collect()
}

/// Retrieves a value from data based on index expressions.
///
/// Evaluates index expressions to determine which element to access.
/// If the validity expression evaluates to non-zero, returns the element at the calculated index.
/// Otherwise, returns zero.
pub(crate) fn get_index(
    data: &[FixedPackedBaseField],
    (ind, val): &(Expression, Expression),
    stack: &mut Vec<i64>,
    index: usize,
) -> FixedPackedBaseField {
    if val.exec_single_var_stack(index, stack) != 0 {
        let i = ind.exec_single_var_stack(index, stack);
        data[i]
    } else {
        FixedPackedBaseField::zero()
    }
}
