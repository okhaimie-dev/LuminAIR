use stwo_prover::constraint_framework::{EvalAtRow, FrameworkComponent, FrameworkEval};

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
        eval.add_constraint(c - (a + b));

        eval
    }
}
