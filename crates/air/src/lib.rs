pub mod gen;
pub mod prover;

/// Represents a claim.
#[derive(Debug, Eq, PartialEq)]
pub struct Claim {
    /// Logarithmic size (`log2`) of the evaluated trace.
    pub log_size: u32,
}
