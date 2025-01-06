use component::{TensorAddComponent, TensorAddEval};
use stwo_prover::{
    constraint_framework::{FrameworkComponent, TraceLocationAllocator},
    core::{
        air::{Component, ComponentProver},
        backend::{Backend, BackendForChannel},
        channel::MerkleChannel,
        fields::m31::BaseField,
        pcs::{CommitmentSchemeProver, CommitmentSchemeVerifier, PcsConfig},
        poly::{
            circle::{CanonicCoset, CircleEvaluation},
            BitReversedOrder,
        },
        prover::{prove, verify, StarkProof, VerificationError},
        vcs::ops::MerkleHasher,
        ColumnVec,
    },
};
use trace::TensorAddTracer;

use super::{
    tensor::{AirTensor, TensorField},
    Circuit,
};

pub mod component;
pub mod trace;

pub struct TensorAddProof<H: MerkleHasher> {
    pub stark_proof: StarkProof<H>,
}

pub struct TensorAdd<'a, F: TensorField> {
    pub a: &'a AirTensor<'a, F>,
    pub b: &'a AirTensor<'a, F>,
    pub log_size: u32,
}

impl<'t, F: TensorField, B: Backend + TensorAddTracer<F>> Circuit<B> for TensorAdd<'t, F>
where
    FrameworkComponent<TensorAddEval>: ComponentProver<B>,
{
    type Component = TensorAddComponent;
    type Proof<'a, H: MerkleHasher> = TensorAddProof<H>;
    type Error = VerificationError;
    type Trace = ColumnVec<CircleEvaluation<B, BaseField, BitReversedOrder>>;

    fn generate_trace(&self) -> Self::Trace {
        let (trace, _c) = B::generate_trace(self.log_size, self.a, self.b);

        trace
    }

    fn prove<'a, MC: MerkleChannel>(
        trace: &Self::Trace,
        config: PcsConfig,
    ) -> (Vec<Self::Component>, Self::Proof<'a, MC::H>)
    where
        B: BackendForChannel<MC>,
    {
        // Precompute twiddles
        let twiddles = B::precompute_twiddles(
            CanonicCoset::new(trace[0].domain.log_size() + 1 + config.fri_config.log_blowup_factor)
                .circle_domain()
                .half_coset,
        );

        // Setup protocol
        let channel = &mut MC::C::default();
        let mut commitment_scheme = CommitmentSchemeProver::<B, MC>::new(config, &twiddles);

        // Create component
        let component = TensorAddComponent::new(
            &mut TraceLocationAllocator::default(),
            TensorAddEval {
                log_size: trace[0].domain.log_size(),
            },
            (Default::default(), None),
        );

        // Commit preprocessing trace
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals([]);
        tree_builder.commit(channel);

        // Commit main trace
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(trace.clone());
        tree_builder.commit(channel);

        // Generate proof
        let stark_proof = prove::<B, MC>(
            &[&component as &dyn stwo_prover::core::air::ComponentProver<B>],
            channel,
            commitment_scheme,
        )
        .unwrap();

        (vec![component], TensorAddProof { stark_proof })
    }

    fn verify<'a, MC: MerkleChannel>(
        components: Vec<Self::Component>,
        proof: Self::Proof<'a, MC::H>,
        config: PcsConfig,
    ) -> Result<(), Self::Error> {
        let channel = &mut MC::C::default();
        let commitment_scheme = &mut CommitmentSchemeVerifier::<MC>::new(config);

        // Get expected column sizes from component
        let log_sizes = components[0].trace_log_degree_bounds();

        // Verify main trace commitment
        commitment_scheme.commit(proof.stark_proof.commitments[0], &log_sizes[0], channel);

        // Verify constant trace commitment
        commitment_scheme.commit(proof.stark_proof.commitments[1], &log_sizes[1], channel);

        // Verify the proof
        verify(
            &components
                .iter()
                .map(|c| c as &dyn stwo_prover::core::air::Component)
                .collect::<Vec<_>>(),
            channel,
            commitment_scheme,
            proof.stark_proof,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    use stwo_prover::core::{
        backend::simd::{
            m31::{PackedBaseField, LOG_N_LANES},
            SimdBackend,
        },
        fields::m31::BaseField,
        vcs::blake2_merkle::Blake2sMerkleChannel,
    };

    #[test]
    fn test_tensor_add_e2e() {
        let config = PcsConfig::default();

        // Test cases with different shapes
        let binding_a_1 = [PackedBaseField::broadcast(BaseField::from_u32_unchecked(1)); 4];
        let binding_b_1 = [PackedBaseField::broadcast(BaseField::from_u32_unchecked(2)); 4];
        let binding_a_2 = [PackedBaseField::broadcast(BaseField::from_u32_unchecked(5))];
        let binding_b_2 = [PackedBaseField::broadcast(BaseField::from_u32_unchecked(1)); 6];
        let binding_a_3 = [PackedBaseField::broadcast(BaseField::from_u32_unchecked(1)); 3];
        let binding_b_3 = [PackedBaseField::broadcast(BaseField::from_u32_unchecked(2)); 6];
        let binding_a_4 = [PackedBaseField::broadcast(BaseField::from_u32_unchecked(3)); 2];
        let binding_b_4 = [PackedBaseField::broadcast(BaseField::from_u32_unchecked(1)); 6];
        let binding_a_5 = [PackedBaseField::broadcast(BaseField::from_u32_unchecked(2)); 3];
        let binding_b_5 = [PackedBaseField::broadcast(BaseField::from_u32_unchecked(3)); 6];
        let test_cases = vec![
            // Case 1: Same shape tensors (2x2)
            (
                AirTensor::new(&binding_a_1, vec![2, 2]),
                AirTensor::new(&binding_b_1, vec![2, 2]),
            ),
            // Case 2: Broadcasting scalar to matrix (1 -> 2x3)
            (
                AirTensor::new(&binding_a_2, vec![1]),
                AirTensor::new(&binding_b_2, vec![2, 3]),
            ),
            // Case 3: Broadcasting row to matrix (1x3 -> 2x3)
            (
                AirTensor::new(&binding_a_3, vec![1, 3]),
                AirTensor::new(&binding_b_3, vec![2, 3]),
            ),
            // Case 4: Broadcasting column to matrix (2x1 -> 2x3)
            (
                AirTensor::new(&binding_a_4, vec![2, 1]),
                AirTensor::new(&binding_b_4, vec![2, 3]),
            ),
            // Case 5: Different rank tensors (1x1x3 -> 2x1x3)
            (
                AirTensor::new(&binding_a_5, vec![1, 1, 3]),
                AirTensor::new(&binding_b_5, vec![2, 1, 3]),
            ),
            // Case 6: Large matrices
            (
                AirTensor::create::<SimdBackend>(
                    (0..50 * 50).map(|i| i as u32).collect(),
                    vec![50, 50],
                ),
                AirTensor::create::<SimdBackend>((0..50).map(|i| i as u32).collect(), vec![50, 1]),
            ),
        ];

        // Run each test case through the trace generation, proving and verification
        for (i, (tensor_a, tensor_b)) in test_cases.into_iter().enumerate() {
            println!("Testing case {}", i + 1);

            // Calculate required log_size based on tensor dimensions
            let max_elements = tensor_a.size().max(tensor_b.size());
            let required_log_size = ((max_elements + (1 << LOG_N_LANES) - 1) >> LOG_N_LANES)
                .next_power_of_two()
                .trailing_zeros()
                + LOG_N_LANES;

            // Create circuit instance
            let circuit = TensorAdd {
                a: &tensor_a,
                b: &tensor_b,
                log_size: required_log_size,
            };

            // Generate trace
            let start_trace = Instant::now();
            let trace = circuit.generate_trace();
            let trace_time = start_trace.elapsed();
            println!("Trace generation time for case {}: {:?}", i + 1, trace_time);

            // Generate proof
            let start_prove = Instant::now();
            let (components, proof) = TensorAdd::prove::<Blake2sMerkleChannel>(&trace, config);
            let prove_time = start_prove.elapsed();
            println!("Proving time for case {}: {:?}", i + 1, prove_time);

            // Verify proof
            let start_verify = Instant::now();
            TensorAdd::verify::<Blake2sMerkleChannel>(components, proof, config)
                .unwrap_or_else(|_| panic!("Verification failed for test case {}", i + 1));
            let verify_time = start_verify.elapsed();
            println!("Verifying time for case {}: {:?}", i + 1, verify_time);
        }
    }
}
