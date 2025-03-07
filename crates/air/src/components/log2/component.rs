use num_traits::{One, Zero};
use numerair::eval::EvalFixedPoint;
use stwo_prover::{
    constraint_framework::{EvalAtRow, FrameworkComponent, FrameworkEval, RelationEntry},
    core::fields::{m31::BaseField, qm31::SecureField},
};

use crate::components::{Log2Claim, NodeElements};

pub type Log2Component = FrameworkComponent<Log2Eval>;

pub struct Log2Eval {
    log_size: u32,
    lookup_elements: NodeElements,
    input_multiplicity: SecureField,
    out_multiplicity: SecureField,
}

impl Log2Eval {
    pub fn new(claim: &Log2Claim, lookup_elements: NodeElements) -> Self {
        let input_multiplicity = if claim.node_info.inputs[0].is_initializer {
            SecureField::zero()
        } else {
            -SecureField::one()
        };

        let out_multiplicity = if claim.node_info.output.is_final_output {
            SecureField::zero()
        } else {
            SecureField::one() * BaseField::from_u32_unchecked(claim.node_info.num_consumers as u32)
        };

        Self {
            log_size: claim.log_size,
            lookup_elements,
            input_multiplicity,
            out_multiplicity,
        }
    }
}

impl FrameworkEval for Log2Eval {
    fn log_size(&self) -> u32 {
        self.log_size
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let input = eval.next_trace_mask();
        let out = eval.next_trace_mask();

        // Add log2 constraint
        eval.eval_fixed_log2(input.clone(), out.clone());

        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            self.input_multiplicity.into(),
            &[input],
        ));

        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            self.out_multiplicity.into(),
            &[out],
        ));

        eval.finalize_logup();
        eval
    }
}
