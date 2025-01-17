use crate::tensor::AirTensor;
use numerair::{eval::EvalFixedPoint, SCALE_FACTOR};
use stwo_prover::{
    constraint_framework::{EvalAtRow, FrameworkComponent, FrameworkEval},
    core::{prover::StarkProof, vcs::ops::MerkleHasher},
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

impl FrameworkEval for TensorMulEval {
    fn log_size(&self) -> u32 {
        self.log_size
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        // Get values from trace
        let lhs = eval.next_trace_mask();
        let rhs = eval.next_trace_mask();
        let out = eval.next_trace_mask();
        let rem = eval.next_trace_mask();

        eval.eval_fixed_mul(lhs, rhs, SCALE_FACTOR.into(), out, rem);

        eval
    }
}
