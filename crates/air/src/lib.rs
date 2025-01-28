#![feature(trait_upcasting)]

use ::serde::{Deserialize, Serialize};
use components::{
    add::{
        components::{AddComponent, AddEval},
        trace::AddElements,
    },
    TraceEval,
};
use pie::ClaimType;
use prover::IS_FIRST_LOG_SIZES;
use stwo_prover::{
    constraint_framework::{
        preprocessed_columns::PreprocessedColumn, TraceLocationAllocator, PREPROCESSED_TRACE_IDX,
    },
    core::{
        air::{Component, ComponentProver},
        backend::simd::SimdBackend,
        channel::Channel,
        pcs::TreeVec,
    },
};

pub mod components;
pub mod pie;
pub mod prover;
pub mod serde;
pub mod utils;

/// A claim over the log sizes for each component of the system.
///
/// A component is made of three types of trace:
/// - Preprocessed Trace (Phase 0)
/// - Main Trace (Phase 1)
/// - Interaction Trace (Phase 2)
#[derive(Serialize, Deserialize, Debug)]
pub struct LuminairClaim {
    pub claims: Vec<ClaimType>,
}

impl LuminairClaim {
    pub fn mix_into(&self, channel: &mut impl Channel) {
        // Mix all Add claims
        for claim in &self.claims {
            claim.mix_into(channel);
        }
    }

    pub fn log_sizes(&self) -> TreeVec<Vec<u32>> {
        // Combine log sizes from all components
        let mut log_sizes = TreeVec::concat_cols(self.claims.iter().map(|claim| claim.log_size()));

        // Overwrite preprocessed column claim
        log_sizes[PREPROCESSED_TRACE_IDX] = IS_FIRST_LOG_SIZES.to_vec();

        log_sizes
    }
}

/// All the interaction elements required by the components during the interaction phase 2.
///
/// The elements are drawn from a Fiat-Shamir [`Channel`], currently using the BLAKE2 hash.
pub struct LuminairInteractionElements {
    pub add_lookup_elements: AddElements,
}

impl LuminairInteractionElements {
    /// Draw all the interaction elements needed for
    /// all the components of the system.
    pub fn draw(channel: &mut impl Channel) -> Self {
        Self {
            add_lookup_elements: AddElements::draw(channel),
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
    pub fn new(claims: &Vec<ClaimType>) -> Self {
        let tree_span_provider = &mut TraceLocationAllocator::new_with_preproccessed_columns(
            &IS_FIRST_LOG_SIZES
                .iter()
                .copied()
                .map(PreprocessedColumn::IsFirst)
                .collect::<Vec<_>>(),
        );

        // Create a component for each claim

        let components = claims
            .iter()
            .map(|claim| match claim {
                ClaimType::Add(c) => AddComponent::new(
                    tree_span_provider,
                    AddEval::new(c),
                    (Default::default(), None),
                ),
            })
            .collect();

        Self { add: components }
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

    /// Returns the `Component` of each components, used by the verifier.
    pub fn components(&self) -> Vec<&dyn Component> {
        self.provers()
            .into_iter()
            .map(|component| component as &dyn Component)
            .collect()
    }
}

pub struct LuminairTrace {
    pub traces: Vec<TraceEval>,
    pub claims: LuminairClaim,
}
