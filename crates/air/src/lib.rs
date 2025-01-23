use components::add::{
    components::{AddComponent, AddEval},
    trace::AddClaim,
};
use prover::IS_FIRST_LOG_SIZES;
use serde::{Deserialize, Serialize};
use stwo_prover::{
    constraint_framework::{preprocessed_columns::PreprocessedColumn, TraceLocationAllocator},
    core::{air::ComponentProver, backend::simd::SimdBackend, channel::Channel},
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
    pub add: Vec<AddClaim>,
}

impl LuminairClaim {
    pub fn mix_into(&self, channel: &mut impl Channel) {
        // Mix all Add claims
        for claim in &self.add {
            claim.mix_into(channel);
        }
    }
}

/// All the components that constitute Luminair.
///
/// Components are used by the prover as a `ComponentProver`,
/// and by the verifier as a `Component`.
pub struct LuminairComponents {
    add: Vec<AddComponent>,
}

impl LuminairComponents {
    pub fn new(claim: &LuminairClaim) -> Self {
        let tree_span_provider = &mut TraceLocationAllocator::new_with_preproccessed_columns(
            &IS_FIRST_LOG_SIZES
                .iter()
                .copied()
                .map(PreprocessedColumn::IsFirst)
                .collect::<Vec<_>>(),
        );

        // Create a component for each Add claim
        let add = claim
            .add
            .iter()
            .map(|add_claim| {
                AddComponent::new(
                    tree_span_provider,
                    AddEval::new(add_claim),
                    (Default::default(), None),
                )
            })
            .collect();

        Self { add }
    }

    /// Returns the `ComponentProver` of each components, used by the prover.
    pub fn provers(&self) -> Vec<&dyn ComponentProver<SimdBackend>> {
        // Collect all component provers into a single vector
        let mut provers = Vec::new();
        // Add each Add component prover
        for add in &self.add {
            provers.push(add as &dyn ComponentProver<SimdBackend>);
        }
        provers
    }
}
