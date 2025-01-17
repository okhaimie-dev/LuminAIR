use crate::tensor::AirTensor;
use numerair::eval::EvalFixedPoint;
use stwo_prover::{
    constraint_framework::{EvalAtRow, FrameworkComponent, FrameworkEval},
    core::{prover::StarkProof, vcs::ops::MerkleHasher},
};

pub mod simd;

pub type TensorAddComponent = FrameworkComponent<TensorAddEval>;

pub struct TensorAddProof<H: MerkleHasher> {
    pub stark_proof: StarkProof<H>,
}

pub struct TensorAdd<'a, F> {
    pub a: &'a AirTensor<'a, F>,
    pub b: &'a AirTensor<'a, F>,
    pub log_size: u32,
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
        eval.eval_fixed_add(a, b, c);

        eval
    }
}
