use numerair::eval::EvalFixedPoint;
use stwo_prover::constraint_framework::{EvalAtRow, FrameworkComponent, FrameworkEval};

use crate::components::AddClaim;

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
}

impl AddEval {
    pub const fn new(claim: &AddClaim) -> Self {
        Self {
            log_size: claim.log_size,
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

        eval.eval_fixed_add(lhs, rhs, out);

        eval
    }
}
