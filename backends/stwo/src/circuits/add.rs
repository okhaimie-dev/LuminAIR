use num_traits::Zero;
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
        channel::MerkleChannel,
        fields::{m31::BaseField, qm31::SecureField},
        pcs::{CommitmentSchemeProver, CommitmentSchemeVerifier, PcsConfig},
        poly::{
            circle::{CanonicCoset, CircleEvaluation, PolyOps},
            BitReversedOrder,
        },
        prover::{prove, verify, StarkProof, VerificationError},
        ColumnVec,
    },
};

// Struct to represent our tensor with SIMD-friendly storage
#[derive(Clone)]
pub struct Tensor {
    data: Vec<PackedBaseField>,
    dims: Vec<usize>,
    stride: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<PackedBaseField>, dims: Vec<usize>) -> Self {
        let stride = Self::compute_stride(&dims);
        Self { data, dims, stride }
    }

    fn compute_stride(dims: &[usize]) -> Vec<usize> {
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
}

// The main circuit component for tensor addition
pub type TensorAddComponent = FrameworkComponent<TensorAddEval>;

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
) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    assert!(a.is_broadcastable_with(&b), "Tensors must be broadcastable");

    // Calculate required trace size
    let max_size = a.size().max(b.size());
    assert!(log_size >= LOG_N_LANES);

    // Initialize trace columns
    let mut trace = vec![
        Col::<SimdBackend, BaseField>::zeros(1 << log_size),
        Col::<SimdBackend, BaseField>::zeros(1 << log_size),
        Col::<SimdBackend, BaseField>::zeros(1 << log_size),
    ];

    // Fill trace with tensor data
    for vec_row in 0..(1 << (log_size - LOG_N_LANES)) {
        if vec_row < max_size {
            // Handle broadcasting by repeating values as needed
            let a_idx = vec_row % a.size();
            let b_idx = vec_row % b.size();

            trace[0].data[vec_row] = a.data[a_idx];
            trace[1].data[vec_row] = b.data[b_idx];
            trace[2].data[vec_row] = a.data[a_idx] + b.data[b_idx];
        }
    }

    let domain = CanonicCoset::new(log_size).circle_domain();
    trace
        .into_iter()
        .map(|eval| CircleEvaluation::new(domain, eval))
        .collect()
}

fn prover<MC: MerkleChannel>(
    log_size: u32,
    config: PcsConfig,
    a: Tensor,
    b: Tensor,
) -> (TensorAddComponent, StarkProof<MC::H>)
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

    // Preprocess Trace
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals([]);
    tree_builder.commit(prover_channel);

    // Generate Trace
    println!("Start generating trace ...");
    let trace = generate_trace(log_size, a, b);
    println!("Trace 0 len: {:?}", trace[0].length);
    println!("Trace 0: {:?}", trace[0]);

    println!("Trace generated!");
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
    println!("Start proving trace ...");
    let proof = prove::<SimdBackend, MC>(&[&component], prover_channel, commitment_scheme).unwrap();
    println!("Proved successfully!");

    (component, proof)
}

fn verifier<MC: MerkleChannel>(
    config: PcsConfig,
    component: TensorAddComponent,
    proof: StarkProof<MC::H>,
) -> Result<(), VerificationError> {
    let channel = &mut MC::C::default();
    let commitment_scheme = &mut CommitmentSchemeVerifier::<MC>::new(config);

    // Get expected column sizes from component
    let log_sizes = component.trace_log_degree_bounds();

    // Verify main trace commitment
    commitment_scheme.commit(proof.commitments[0], &log_sizes[0], channel);

    // Verify constant trace commitment
    commitment_scheme.commit(proof.commitments[1], &log_sizes[1], channel);

    // Verify the proof
    verify(&[&component], channel, commitment_scheme, proof)
}

#[cfg(test)]
mod tests {
    use super::*;
    use stwo_prover::{
        constraint_framework::{assert_constraints, FrameworkEval},
        core::{pcs::TreeVec, vcs::blake2_merkle::Blake2sMerkleChannel},
    };

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
    fn test_tensor_add_different_shapes() {
        const LOG_SIZE: u32 = 8; // Increased to handle larger tensors
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
        ];

        // Run each test case through the prover and verifier
        for (i, (tensor_a, tensor_b)) in test_cases.into_iter().enumerate() {
            println!("Testing case {}", i + 1);
            let (component, proof) =
                prover::<Blake2sMerkleChannel>(LOG_SIZE, config, tensor_a, tensor_b);
            verifier::<Blake2sMerkleChannel>(config, component, proof)
                .unwrap_or_else(|_| panic!("Verification failed for test case {}", i + 1));
        }
    }

    #[test]
    fn test_large_tensor_add() {
        // Create large tensors with different shapes
        // Shape: 10 x 10
        let large_matrix_a = {
            let n = 10;
            let data = (0..n * n)
                .map(|i| PackedBaseField::broadcast(BaseField::from_u32_unchecked(i as u32 % 1000)))
                .collect();
            Tensor::new(data, vec![n, n])
        };

        // Shape: 10 x 1 (broadcasting column)
        let column_b = {
            let n = 10;
            let data = (0..n)
                .map(|i| PackedBaseField::broadcast(BaseField::from_u32_unchecked(i as u32 % 100)))
                .collect();
            Tensor::new(data, vec![n, 1])
        };

        // Calculate required log_size based on tensor dimensions
        let max_elements = large_matrix_a.size().max(column_b.size());
        // Need to account for SIMD lanes and round up to next power of 2
        let required_log_size = (max_elements * LOG_N_LANES as usize)
            .next_power_of_two()
            .trailing_zeros();

        println!(
            "Using log_size = {} for {} elements",
            required_log_size, max_elements
        );
        let config = PcsConfig::default();

        println!("Starting proving process for large tensors...");
        let start = std::time::Instant::now();

        // Run proof generation
        let (component, proof) =
            prover::<Blake2sMerkleChannel>(required_log_size, config, large_matrix_a, column_b);

        let proving_time = start.elapsed();
        println!("Proving completed in {:?}", proving_time);

        // Run verification
        let verify_start = std::time::Instant::now();
        verifier::<Blake2sMerkleChannel>(config, component, proof).unwrap();
        let verify_time = verify_start.elapsed();

        println!("Verification completed in {:?}", verify_time);
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
}
