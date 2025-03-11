use numerair::eval::EvalFixedPoint;
use stwo_prover::constraint_framework::{
    EvalAtRow, FrameworkComponent, FrameworkEval, RelationEntry,
};

use crate::components::{AddClaim, NodeElements};

/// Component for addition operations, using `SimdBackend` with fallback to `CpuBackend` for small traces.
pub type AddComponent = FrameworkComponent<AddEval>;

/// Defines the AIR for the addition component.
pub struct AddEval {
    log_size: u32,
    lookup_elements: NodeElements,
}

impl AddEval {
    /// Creates a new `AddEval` instance from a claim and lookup elements.
    pub fn new(claim: &AddClaim, lookup_elements: NodeElements) -> Self {
        Self {
            log_size: claim.log_size,
            lookup_elements,
        }
    }
}

impl FrameworkEval for AddEval {
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
        let lhs_mult = eval.next_trace_mask();
        let rhs_mult = eval.next_trace_mask();
        let out_mult = eval.next_trace_mask();

        // ┌─────────────────────────────┐
        // │   Consistency Constraints   │
        // └─────────────────────────────┘

        eval.eval_fixed_add(lhs.clone(), rhs.clone(), out.clone());

        // ┌─────────────────────────────┐
        // │   Interaction Constraints   │
        // └─────────────────────────────┘

        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            lhs_mult.into(),
            &[lhs],
        ));

        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            rhs_mult.into(),
            &[rhs],
        ));

        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            out_mult.into(),
            &[out],
        ));

        eval.finalize_logup();

        eval
    }
}
