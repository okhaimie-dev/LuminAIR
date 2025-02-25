use crate::components::{MulClaim, NodeElements};
use num_traits::{One, Zero};
use numerair::{eval::EvalFixedPoint, SCALE_FACTOR};
use stwo_prover::{
    constraint_framework::{EvalAtRow, FrameworkComponent, FrameworkEval, RelationEntry},
    core::fields::{m31::BaseField, qm31::SecureField},
};

/// Component for multiplication operations, using `SimdBackend` with fallback to `CpuBackend` for small traces.
pub type MulComponent = FrameworkComponent<MulEval>;

/// Defines the AIR for the multiplication component.
pub struct MulEval {
    log_size: u32,
    lookup_elements: NodeElements,
    lhs_multiplicity: SecureField,
    rhs_multiplicity: SecureField,
    out_multiplicity: SecureField,
}

impl MulEval {
    /// Creates a new `MulEval` instance from a claim and lookup elements.
    pub fn new(claim: &MulClaim, lookup_elements: NodeElements) -> Self {
        let lhs_multiplicity = if claim.node_info.inputs[0].is_initializer {
            SecureField::zero()
        } else {
            -SecureField::one()
        };

        let rhs_multiplicity = if claim.node_info.inputs[1].is_initializer {
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
            lhs_multiplicity,
            rhs_multiplicity,
            out_multiplicity,
        }
    }
}

impl FrameworkEval for MulEval {
    /// Returns the logarithmic size of the main trace.
    fn log_size(&self) -> u32 {
        self.log_size
    }

    /// The degree of the constraints is bounded by the size of the trace.
    ///
    /// Returns the ilog2 (upper) bound of the constraint degree for the component.
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size + 1
    }

    /// Evaluates the AIR constraints for the addition operation.
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let lhs = eval.next_trace_mask();
        let rhs = eval.next_trace_mask();
        let out = eval.next_trace_mask();
        let rem = eval.next_trace_mask();

        eval.eval_fixed_mul(
            lhs.clone(),
            rhs.clone(),
            SCALE_FACTOR.into(),
            out.clone(),
            rem,
        );

        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            self.lhs_multiplicity.into(),
            &[lhs],
        ));

        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            self.rhs_multiplicity.into(),
            &[rhs],
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
