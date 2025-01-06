use stwo_prover::core::backend::simd::m31::LOG_N_LANES;

pub fn calculate_log_size(max_size: usize, ) -> u32{
    ((max_size + (1 << LOG_N_LANES) - 1) >> LOG_N_LANES)
    .next_power_of_two()
    .trailing_zeros() + LOG_N_LANES
}