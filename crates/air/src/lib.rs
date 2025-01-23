use components::{
    add::{
        components::{AddComponent, AddEval},
        trace::AddClaim,
    },
    Claim,
};
use prover::IS_FIRST_LOG_SIZES;
use serde::{Deserialize, Serialize};
use stwo_prover::{
    constraint_framework::{preprocessed_columns::PreprocessedColumn, TraceLocationAllocator},
    core::channel::Channel,
};

pub mod components;
pub mod prover;
pub mod utils;

/// A claim over the log sizes for each component of the system.
///
/// A component is made of three types of trace:
/// - Preprocessed Trace (Phase 0)
/// - Main Trace (Phase 1)
/// - Interaction Trace (Phase 2)
#[derive(Serialize, Deserialize, Debug)]
pub struct LuminairClaim {
    pub add: AddClaim,
}

impl LuminairClaim {
    pub fn mix_into(&self, channel: &mut impl Channel) {
        self.add.mix_into(channel);
    }
}

/// All the components that constitute Luminair.
///
/// Components are used by the prover as a `ComponentProver`,
/// and by the verifier as a `Component`.
pub struct LuminairComponents {
    add: AddComponent,
}

impl LuminairComponents {
    /// Initilizes all luminair components from the claims generated from the trace.
    pub fn new(claim: &LuminairClaim) -> Self {
        let tree_span_provider = &mut TraceLocationAllocator::new_with_preproccessed_columns(
            &IS_FIRST_LOG_SIZES
                .iter()
                .copied()
                .map(PreprocessedColumn::IsFirst)
                .collect::<Vec<_>>(),
        );

        let add = AddComponent::new(
            tree_span_provider,
            AddEval::new(&claim.add),
            (Default::default(), None),
        );

        Self { add }
    }
}
