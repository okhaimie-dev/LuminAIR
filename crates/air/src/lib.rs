use stwo_prover::core::{
    backend::{Backend, BackendForChannel},
    channel::MerkleChannel,
    pcs::PcsConfig,
    vcs::ops::MerkleHasher,
};

pub mod ops;
pub mod tensor;
pub mod utils;
pub mod serde;

pub trait Circuit<B: Backend> {
    type Component;
    type Proof<'a, H: MerkleHasher>;
    type Error;
    type Trace;

    /// Generates the execution trace for the circuit
    fn generate_trace(&self) -> Self::Trace;

    /// Creates proof for a given trace
    fn prove<'a, MC: MerkleChannel>(
        trace: &Self::Trace,
        config: PcsConfig,
    ) -> (Vec<Self::Component>, Self::Proof<'a, MC::H>)
    where
        B: BackendForChannel<MC>;

    /// Verifies a proof
    fn verify<'a, MC: MerkleChannel>(
        components: Vec<Self::Component>,
        proof: Self::Proof<'a, MC::H>,
        config: PcsConfig,
    ) -> Result<(), Self::Error>;
}
