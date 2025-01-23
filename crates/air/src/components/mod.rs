use serde::{Deserialize, Serialize};
use stwo_prover::core::{
    backend::simd::SimdBackend,
    channel::Channel,
    fields::m31::BaseField,
    poly::{circle::CircleEvaluation, BitReversedOrder},
    ColumnVec,
};

pub mod add;

/// Type for trace evaluation to be used in Stwo.
pub type TraceEval = ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>;

/// Represents a claim.
#[derive(Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Claim {
    /// Logarithmic size (`log2`) of the evaluated trace.
    pub log_size: u32,
}

impl Claim {
    /// Creates a new claim.
    pub const fn new(log_size: u32) -> Self {
        Self { log_size }
    }

    /// Mix the log size of the table to the Fiat-Shamir [`Channel`],
    /// to bound the channel randomness and the trace.
    pub fn mix_into(&self, channel: &mut impl Channel) {
        channel.mix_u64(self.log_size.into());
    }
}
