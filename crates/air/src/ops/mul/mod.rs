use crate::tensor::AirTensor;
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
        let a = eval.next_trace_mask();
        let b = eval.next_trace_mask();
        let c = eval.next_trace_mask();

        // Mul constraint: c = a * b
        eval.add_constraint(c - (a * b));

        eval
    }
}
