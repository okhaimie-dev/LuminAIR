use crate::tensor::AirTensor;
use lazy_static::lazy_static;
use numerair::fixed_points::{DEFAULT_SCALE, MAX_SCALE};
use stwo_prover::{
    constraint_framework::{EvalAtRow, FrameworkComponent, FrameworkEval},
    core::{fields::m31::M31, prover::StarkProof, vcs::ops::MerkleHasher},
};

pub mod simd;

pub type TensorMulComponent = FrameworkComponent<TensorMulEval>;

pub struct TensorMulProof<H: MerkleHasher> {
    pub stark_proof: StarkProof<H>,
}

pub struct TensorMul<'a, F> {
    pub a: &'a AirTensor<'a, F>,
    pub b: &'a AirTensor<'a, F>,
    pub log_size: u32,
}

#[derive(Clone)]
pub struct TensorMulEval {
    pub log_size: u32,
}

// Constants computed at compile time
const SCALE_FACTOR_RAW: u32 = 1u32 << DEFAULT_SCALE;
const MAX_VAL_RAW: u32 = (1u32 << (MAX_SCALE - 1)) - 1;

lazy_static! {
    // Lazily computed field elements that are reused
    static ref SCALE_FACTOR: M31 = M31::from_u32_unchecked(SCALE_FACTOR_RAW);
    static ref SCALE_FACTOR_INV: M31 = SCALE_FACTOR.inverse();
    static ref MAX_VAL: M31 = M31::from_u32_unchecked(MAX_VAL_RAW);
    static ref MIN_VAL: M31 = -(*MAX_VAL);
}

impl FrameworkEval for TensorMulEval {
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

        // For fixed point multiplication we need to:
        // 1. Multiply the values
        // 2. Divide by the scale factor to get back to the right fixed point

        // Handle the division by implementing:
        // c * 2^SCALE = a * b

        // Main constraint: c * scale = a * b
        eval.add_constraint(c.clone() * E::F::from(*SCALE_FACTOR) - (a.clone() * b.clone()));

        // TODO (@raphaeDkhn): Range checks to enforce min_val ≤ x ≤ max_val

        eval
    }
}
