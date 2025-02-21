use num_traits::{One, Zero};
use numerair::eval::EvalFixedPoint;
use stwo_prover::{
    constraint_framework::{EvalAtRow, FrameworkComponent, FrameworkEval, RelationEntry},
    core::fields::m31::BaseField,
};

use crate::{components::AddClaim, pie::NodeInfo};

use super::table::AddElements;

/// Implementation of `Component` and `ComponentProver` for the Add component.
/// It targets the `SimdBackend` from the Stwo constraint framework, with the fallback
/// on `CpuBackend` for small traces.
pub type AddComponent = FrameworkComponent<AddEval>;

/// The AIR for the [`AddComponent`]
///
/// Constraints are defined though the [`FrameworkEval`]
/// provided by the constraint framework of Stwo.
pub struct AddEval {
    /// The log size of the component's main trace height.
    log_size: u32,
    /// The random elements used for the lookup protocol.
    lookup_elements: AddElements,
    /// Node information.
    node_info: NodeInfo,
}

impl AddEval {
    pub fn new(claim: &AddClaim, lookup_elements: AddElements) -> Self {
        Self {
            log_size: claim.log_size,
            node_info: claim.node_info.clone(),
            lookup_elements,
        }
    }
}

impl FrameworkEval for AddEval {
    /// Returns the log size from the main claim.
    fn log_size(&self) -> u32 {
        self.log_size
    }

    /// The degree of the constraints is bounded by the size of the trace.
    ///
    /// Returns the ilog2 (upper) bound of the constraint degree for the component.
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size + 1
    }

    /// Defines the AIR for the Add component.
    ///
    /// Values from the current row are obtained through masks.
    /// When you apply a mask, you target the current column and then pass to the next
    /// one: the register order matters to correctly fetch them, and all registers must be fetched.
    ///
    /// - Use `eval.next_trace_mask()` to get the current register from the main trace
    ///   (`ORIGINAL_TRACE_IDX`)
    ///
    /// Use `eval.add_constraint` to define a local constraint (boundary, consistency, transition).
    /// Use `eval.add_to_relation` to define a global constraint for the logUp protocol.
    ///
    /// The logUp must be finalized with `eval.finalize_logup()`.
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let lhs = eval.next_trace_mask();
        let rhs = eval.next_trace_mask();
        let out = eval.next_trace_mask();

        eval.eval_fixed_add(lhs.clone(), rhs.clone(), out.clone());

        let lhs_multiplicity = if self.node_info.inputs[0].is_initializer {
            E::EF::zero()
        } else {
            -E::EF::one()
        };

        let rhs_multiplicity = if self.node_info.inputs[1].is_initializer {
            E::EF::zero()
        } else {
            -E::EF::one()
        };
        let out_multiplicity = if self.node_info.output.is_final_output {
            E::EF::zero()
        } else {
            E::EF::one() * BaseField::from_u32_unchecked(self.node_info.num_consumers as u32)
        };

        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            lhs_multiplicity,
            &[lhs],
        ));

        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            rhs_multiplicity,
            &[rhs],
        ));

        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            out_multiplicity,
            &[out],
        ));

        eval.finalize_logup();

        eval
    }
}
