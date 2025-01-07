use stwo_prover::core::{
    backend::{Backend, BackendForChannel},
    channel::MerkleChannel,
    pcs::PcsConfig,
    vcs::ops::MerkleHasher,
};
use tensor::AirTensor;

pub mod ops;
pub mod serde;
pub mod tensor;
pub mod utils;

pub trait Circuit<B: Backend> {
    type Component;
    type Proof<'a, H: MerkleHasher>;
    type Error;
    type Trace;
    type Field;

    /// Generates the execution trace and output tensor for the circuit
    fn generate_trace(&self) -> (Self::Trace, AirTensor<'static, Self::Field>);

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
