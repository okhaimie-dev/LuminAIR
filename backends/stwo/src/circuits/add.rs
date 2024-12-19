use num_traits::Zero;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use stwo_prover::{
    constraint_framework::{EvalAtRow, FrameworkComponent, FrameworkEval, TraceLocationAllocator},
    core::{
        air::Component,
        backend::{
            simd::{
                m31::{PackedBaseField, LOG_N_LANES},
                SimdBackend,
            },
            BackendForChannel, Col, Column,
        },
        channel::{Channel, MerkleChannel},
        fields::{m31::BaseField, qm31::SecureField},
        pcs::{CommitmentSchemeProver, CommitmentSchemeVerifier, PcsConfig},
        poly::{
            circle::{CanonicCoset, CircleEvaluation, PolyOps},
            BitReversedOrder,
        },
        prover::{prove, verify, StarkProof, VerificationError},
        vcs::ops::MerkleHasher,
        ColumnVec,
    },
};

#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: Vec<PackedBaseField>,
    pub dims: Vec<usize>,
    pub stride: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<PackedBaseField>, dims: Vec<usize>) -> Self {
        let stride = Self::compute_stride(&dims);
        Self { data, dims, stride }
    }

    pub fn compute_stride(dims: &[usize]) -> Vec<usize> {
        let mut stride = vec![1; dims.len()];
        for i in (0..dims.len() - 1).rev() {
            stride[i] = stride[i + 1] * dims[i + 1];
        }
        stride
    }

    // Get total number of elements
    pub fn size(&self) -> usize {
        self.dims.iter().product()
    }

    // Check if tensors are broadcastable
    pub fn is_broadcastable_with(&self, other: &Tensor) -> bool {
        let max_dims = self.dims.len().max(other.dims.len());
        let pad_self = max_dims - self.dims.len();
        let pad_other = max_dims - other.dims.len();

        for i in 0..max_dims {
            let dim_self = if i < pad_self {
                1
            } else {
                self.dims[i - pad_self]
            };
            let dim_other = if i < pad_other {
                1
            } else {
                other.dims[i - pad_other]
            };

            if dim_self != dim_other && dim_self != 1 && dim_other != 1 {
                return false;
            }
        }
        true
    }

    // helper function to create SIMD-efficient packed data
    pub fn pack_data(data: Vec<u32>, dims: &[usize]) -> Vec<PackedBaseField> {
        let total_size = dims.iter().product::<usize>();
        let n_packed = (total_size + (1 << LOG_N_LANES) - 1) >> LOG_N_LANES;

        (0..n_packed)
            .into_par_iter()
            .map(|chunk_idx| {
                let start = chunk_idx << LOG_N_LANES;
                let mut lane_values = [0u32; 1 << LOG_N_LANES];

                for (i, lane) in lane_values.iter_mut().enumerate() {
                    let data_idx = start + i;
                    *lane = if data_idx < data.len() {
                        data[data_idx] % 1000
                    } else {
                        0
                    };
                }

                PackedBaseField::from_array(lane_values.map(|x| BaseField::from_u32_unchecked(x)))
            })
            .collect()
    }
}

// The main circuit component for tensor addition
pub type TensorAddComponent = FrameworkComponent<TensorAddEval>;

#[derive(Clone, Debug)]
pub struct TensorAddPublicInputs {
    pub c: Tensor, // Result tensor as public input
}

pub struct TensorAddProof<H: MerkleHasher> {
    pub public_inputs: TensorAddPublicInputs,
    pub stark_proof: StarkProof<H>,
}

#[derive(Clone)]
pub struct TensorAddEval {
    pub log_size: u32,
}

impl FrameworkEval for TensorAddEval {
    fn log_size(&self) -> u32 {
        self.log_size
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        // Get values from trace
        let a = eval.next_trace_mask();
        let b = eval.next_trace_mask();
        let c = eval.next_trace_mask();

        // Add constraint: c = a + b
        eval.add_constraint(c.clone() - (a + b));

        eval
    }
}

pub fn generate_trace(
    log_size: u32,
    a: Tensor,
    b: Tensor,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    Tensor,
) {
    assert!(a.is_broadcastable_with(&b), "Tensors must be broadcastable");

    // Calculate required trace size
    let max_size = a.size().max(b.size());
    assert!(log_size >= LOG_N_LANES);

    // Initialize trace columns

    let trace_size = 1 << log_size;
    let mut trace: Vec<Col<SimdBackend, BaseField>> = Vec::with_capacity(3);
    let mut c_data = Vec::with_capacity((max_size + (1 << LOG_N_LANES) - 1) >> LOG_N_LANES);
    for _ in 0..3 {
        trace.push(Col::<SimdBackend, BaseField>::zeros(trace_size));
    }

    // Calculate number of SIMD-packed rows needed for each tensor
    let n_rows = 1 << (log_size - LOG_N_LANES);
    let a_packed_size = (a.size() + (1 << LOG_N_LANES) - 1) >> LOG_N_LANES;
    let b_packed_size = (b.size() + (1 << LOG_N_LANES) - 1) >> LOG_N_LANES;

    // Fill trace with tensor data
    // Process in chunks for better cache utilization
    const CHUNK_SIZE: usize = 64;
    for chunk in (0..n_rows).step_by(CHUNK_SIZE) {
        let end = (chunk + CHUNK_SIZE).min(n_rows);

        for vec_row in chunk..end {
            if vec_row < max_size {
                // Calculate the packed indices with broadcasting
                let a_idx = vec_row % a_packed_size;
                let b_idx = vec_row % b_packed_size;

                let sum = a.data[a_idx] + b.data[b_idx];

                trace[0].data[vec_row] = a.data[a_idx];
                trace[1].data[vec_row] = b.data[b_idx];
                trace[2].data[vec_row] = sum;

                c_data.push(sum);
            }
        }
    }

    // Create output tensor C
    let c = Tensor {
        data: c_data,
        dims: if a.size() > b.size() {
            a.dims.clone()
        } else {
            b.dims.clone()
        },
        stride: Tensor::compute_stride(if a.size() > b.size() {
            &a.dims
        } else {
            &b.dims
        }),
    };

    let domain = CanonicCoset::new(log_size).circle_domain();

    (
        trace
            .into_iter()
            .map(|eval| CircleEvaluation::new(domain, eval))
            .collect(),
        c,
    )
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
    use stwo_prover::core::vcs::blake2_merkle::Blake2sMerkleChannel;

    #[test]
    fn test_tensor_broadcasting() {
        let a = Tensor::new(
            vec![PackedBaseField::broadcast(BaseField::from_u32_unchecked(1)); 2],
            vec![2, 1],
        );
        let b = Tensor::new(
            vec![PackedBaseField::broadcast(BaseField::from_u32_unchecked(2)); 3],
            vec![1, 3],
        );
        assert!(a.is_broadcastable_with(&b));
    }

    #[test]
    #[should_panic(expected = "Tensors must be broadcastable")]
    fn test_non_broadcastable_tensors() {
        const LOG_SIZE: u32 = 4;
        let config = PcsConfig::default();

        // Create tensors with incompatible shapes (2x3 and 3x2)
        let a = Tensor::new(
            vec![PackedBaseField::broadcast(BaseField::from_u32_unchecked(1)); 6],
            vec![2, 3],
        );
        let b = Tensor::new(
            vec![PackedBaseField::broadcast(BaseField::from_u32_unchecked(2)); 6],
            vec![3, 2],
        );

        // This should panic due to non-broadcastable shapes
        let _ = prover::<Blake2sMerkleChannel>(LOG_SIZE, config, a, b);
    }

    #[test]
    fn test_tensor_add_different_shapes() {
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
