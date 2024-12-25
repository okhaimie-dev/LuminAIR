use num_traits::Zero;
use stwo_prover::{
    constraint_framework::TraceLocationAllocator,
    core::{
        air::Component,
        backend::{simd::SimdBackend, BackendForChannel},
        channel::{Channel, MerkleChannel},
        fields::qm31::SecureField,
        pcs::{CommitmentSchemeProver, CommitmentSchemeVerifier, PcsConfig},
        poly::circle::{CanonicCoset, PolyOps},
        prover::{prove, verify, StarkProof, VerificationError},
        vcs::ops::MerkleHasher,
    },
};

use crate::circuits::Tensor;

use super::{
    component::{TensorAddComponent, TensorAddEval},
    trace::generate_trace,
};

#[derive(Clone, Debug)]
pub struct TensorAddPublicInputs {
    pub c: Tensor, // Result tensor as public input
}

pub struct TensorAddProof<H: MerkleHasher> {
    pub public_inputs: TensorAddPublicInputs,
    pub stark_proof: StarkProof<H>,
}

pub fn prover<MC: MerkleChannel>(
    log_size: u32,
    config: PcsConfig,
    a: Tensor,
    b: Tensor,
) -> (TensorAddComponent, TensorAddProof<MC::H>)
where
    SimdBackend: BackendForChannel<MC>,
{
    // Precompute twiddles for FFT operations
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(log_size + 1 + config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );

    // Setup protocol
    let prover_channel = &mut MC::C::default();
    let mut commitment_scheme = CommitmentSchemeProver::<SimdBackend, MC>::new(config, &twiddles);

    // Generate trace and get output tensor C
    let (trace, c) = generate_trace(log_size, a, b);

    // Mix public outputs into channel
    for value in &c.data {
        for base_value in value.to_array() {
            prover_channel.mix_felts(&[SecureField::from(base_value)]);
        }
    }

    // Commit preprocessing trace
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals([]);
    tree_builder.commit(prover_channel);

    // Commit trace
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(trace);
    tree_builder.commit(prover_channel);

    // Create Component
    let component = TensorAddComponent::new(
        &mut TraceLocationAllocator::default(),
        TensorAddEval { log_size },
        (SecureField::zero(), None),
    );

    // Generate Proof
    let stark_proof =
        prove::<SimdBackend, MC>(&[&component], prover_channel, commitment_scheme).unwrap();

    (
        component,
        TensorAddProof {
            public_inputs: TensorAddPublicInputs { c },
            stark_proof,
        },
    )
}

pub fn verifier<MC: MerkleChannel>(
    config: PcsConfig,
    component: TensorAddComponent,
    proof: TensorAddProof<MC::H>,
) -> Result<(), VerificationError> {
    let channel = &mut MC::C::default();
    let commitment_scheme = &mut CommitmentSchemeVerifier::<MC>::new(config);

    // Mix the public inputs into the channel
    for value in &proof.public_inputs.c.data {
        for base_value in value.to_array() {
            channel.mix_felts(&[SecureField::from(base_value)]);
        }
    }

    // Get expected column sizes from component
    let log_sizes = component.trace_log_degree_bounds();

    // Verify main trace commitment
    commitment_scheme.commit(proof.stark_proof.commitments[0], &log_sizes[0], channel);

    // Verify constant trace commitment
    commitment_scheme.commit(proof.stark_proof.commitments[1], &log_sizes[1], channel);

    // Verify the proof
    verify(&[&component], channel, commitment_scheme, proof.stark_proof)
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    use stwo_prover::core::{
        backend::simd::m31::{PackedBaseField, LOG_N_LANES},
        fields::m31::BaseField,
        vcs::blake2_merkle::Blake2sMerkleChannel,
    };

    #[test]
    fn test_tensor_add_e2e() {
        let config = PcsConfig::default();

        // Test cases with different shapes
        let test_cases = vec![
            // Case 1: Same shape tensors (2x2)
            (
                Tensor::new(
                    vec![PackedBaseField::broadcast(BaseField::from_u32_unchecked(1)); 4],
                    vec![2, 2],
                ),
                Tensor::new(
                    vec![PackedBaseField::broadcast(BaseField::from_u32_unchecked(2)); 4],
                    vec![2, 2],
                ),
            ),
            // Case 2: Broadcasting scalar to matrix (1 -> 2x3)
            (
                Tensor::new(
                    vec![PackedBaseField::broadcast(BaseField::from_u32_unchecked(5))],
                    vec![1],
                ),
                Tensor::new(
                    vec![PackedBaseField::broadcast(BaseField::from_u32_unchecked(1)); 6],
                    vec![2, 3],
                ),
            ),
            // Case 3: Broadcasting row to matrix (1x3 -> 2x3)
            (
                Tensor::new(
                    vec![PackedBaseField::broadcast(BaseField::from_u32_unchecked(1)); 3],
                    vec![1, 3],
                ),
                Tensor::new(
                    vec![PackedBaseField::broadcast(BaseField::from_u32_unchecked(2)); 6],
                    vec![2, 3],
                ),
            ),
            // Case 4: Broadcasting column to matrix (2x1 -> 2x3)
            (
                Tensor::new(
                    vec![PackedBaseField::broadcast(BaseField::from_u32_unchecked(3)); 2],
                    vec![2, 1],
                ),
                Tensor::new(
                    vec![PackedBaseField::broadcast(BaseField::from_u32_unchecked(1)); 6],
                    vec![2, 3],
                ),
            ),
            // Case 5: Different rank tensors (1x1x3 -> 2x1x3)
            (
                Tensor::new(
                    vec![PackedBaseField::broadcast(BaseField::from_u32_unchecked(2)); 3],
                    vec![1, 1, 3],
                ),
                Tensor::new(
                    vec![PackedBaseField::broadcast(BaseField::from_u32_unchecked(3)); 6],
                    vec![2, 1, 3],
                ),
            ),
            // Case 6: Large matrices
            (
                Tensor::new(
                    Tensor::pack_data((0..50 * 50).map(|i| i as u32).collect(), &[50, 50]),
                    vec![50, 50],
                ),
                Tensor::new(
                    Tensor::pack_data((0..50).map(|i| i as u32).collect(), &[50, 1]),
                    vec![50, 1],
                ),
            ),
        ];

        // Run each test case through the prover and verifier
        for (i, (tensor_a, tensor_b)) in test_cases.into_iter().enumerate() {
            println!("Testing case {}", i + 1);

            // Calculate required log_size based on tensor dimensions
            let max_elements = tensor_a.size().max(tensor_b.size());
            let required_log_size = ((max_elements + (1 << LOG_N_LANES) - 1) >> LOG_N_LANES)
                .next_power_of_two()
                .trailing_zeros()
                + LOG_N_LANES;

            let start_prove = Instant::now();
            let (component, proof) =
                prover::<Blake2sMerkleChannel>(required_log_size, config, tensor_a, tensor_b);
            let prove_time = start_prove.elapsed();
            println!("Proving time for case {}: {:?}", i + 1, prove_time);

            let start_verify = Instant::now();
            verifier::<Blake2sMerkleChannel>(config, component, proof)
                .unwrap_or_else(|_| panic!("Verification failed for test case {}", i + 1));
            let verify_time = start_verify.elapsed();
            println!("Verifying time for case {}: {:?}", i + 1, verify_time);
        }
    }
}
