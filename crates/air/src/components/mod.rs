use add::{
    component::{AddComponent, AddEval},
    table::AddColumn,
};
use serde::{Deserialize, Serialize};
use stwo_prover::{
    constraint_framework::{preprocessed_columns::IsFirst, TraceLocationAllocator},
    core::{
        air::{Component, ComponentProver},
        backend::simd::SimdBackend,
        channel::Channel,
        fields::{m31::BaseField, qm31::SecureField, secure_column::SECURE_EXTENSION_DEGREE},
        pcs::TreeVec,
        poly::{circle::CircleEvaluation, BitReversedOrder},
        ColumnVec,
    },
    relation,
};
use thiserror::Error;

use crate::{pie::NodeInfo, LuminairClaim, LuminairInteractionClaim};

pub mod add;

/// Errors related to trace operations.
#[derive(Debug, Error, Eq, PartialEq)]
pub enum TraceError {
    /// The component trace is empty.
    #[error("The trace is empty.")]
    EmptyTrace,
}

/// Alias for trace evaluation columns used in Stwo.
pub type TraceEval = ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>;

/// Claim for the Add trace.
pub type AddClaim = Claim<AddColumn>;

/// Represents columns of a trace.
pub trait TraceColumn {
    /// Returns the number of columns associated with the specific trace type.
    ///
    /// Main trace columns: first element of the tuple
    /// Interaction trace columns: second element of the tuple
    fn count() -> (usize, usize);
}

/// Represents a claim.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Claim<T: TraceColumn> {
    /// Logarithmic size (base 2) of the trace.
    pub log_size: u32,
    /// Information about the node in the computational graph.
    pub node_info: NodeInfo,
    /// Phantom data to associate with the trace column type.
    _marker: std::marker::PhantomData<T>,
}

impl<T: TraceColumn> Claim<T> {
    /// Creates a new claim with the given log size and node information.
    pub const fn new(log_size: u32, node_info: NodeInfo) -> Self {
        Self {
            log_size,
            node_info,
            _marker: std::marker::PhantomData,
        }
    }

    /// Computes log sizes for preprocessed, main, and interaction traces.
    pub fn log_sizes(&self) -> TreeVec<Vec<u32>> {
        let (main_trace_cols, interaction_trace_cols) = T::count();
        let preprocessed_trace_log_sizes: Vec<u32> = vec![self.log_size];
        let trace_log_sizes = vec![self.log_size; main_trace_cols];
        let interaction_trace_log_sizes: Vec<u32> =
            vec![self.log_size; SECURE_EXTENSION_DEGREE * interaction_trace_cols];
        TreeVec::new(vec![
            preprocessed_trace_log_sizes,
            trace_log_sizes,
            interaction_trace_log_sizes,
        ])
    }

    /// Mix the log size of the table and the node structure to the Fiat-Shamir [`Channel`].
    pub fn mix_into(&self, channel: &mut impl Channel) {
        // Mix log_size
        channel.mix_u64(self.log_size.into());

        // Mix number of inputs
        channel.mix_u64(self.node_info.inputs.len() as u64);

        // Mix input flags
        for input in &self.node_info.inputs {
            channel.mix_u64(if input.is_initializer { 1 } else { 0 });
        }

        // Mix output flag
        channel.mix_u64(if self.node_info.output.is_final_output {
            1
        } else {
            0
        });

        // Mix consumer count
        channel.mix_u64(self.node_info.num_consumers as u64);
    }
}

/// Enum representing different types of claims.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum ClaimType {
    Add(Claim<AddColumn>),
}

/// The claim of the interaction phase 2 (with the logUp protocol).
///
/// The claimed sum is the total sum, which is the computed sum of the logUp extension column,
/// including the padding rows.
/// It allows proving that the main trace of a component is either a permutation, or a sublist of
/// another.
#[derive(Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct InteractionClaim {
    /// The computed sum of the logUp extension column, including padding rows (which are actually
    /// set to a multiplicity of 0).
    pub claimed_sum: SecureField,
}

impl InteractionClaim {
    /// Mix the sum from the logUp protocol into the Fiat-Shamir [`Channel`],
    /// to bound the proof to the trace.
    pub fn mix_into(&self, channel: &mut impl Channel) {
        channel.mix_felts(&[self.claimed_sum]);
    }
}

// Defines the relation for the node lookup elements.
// It allows to constrain relationship between nodes.
relation!(NodeElements, 1);

/// All the interaction elements required by the components during the interaction phase 2.
///
/// The elements are drawn from a Fiat-Shamir [`Channel`], currently using the BLAKE2 hash.
#[derive(Clone, Debug)]
pub struct LuminairInteractionElements {
    pub node_lookup_elements: NodeElements,
}

impl LuminairInteractionElements {
    /// Draw all the interaction elements needed for
    /// all the components of the system.
    pub fn draw(channel: &mut impl Channel) -> Self {
        let node_lookup_elements = NodeElements::draw(channel);

        Self {
            node_lookup_elements,
        }
    }
}

/// All the components that consitute LuminAIR.
///
/// Components are used by the prover as a `ComponentProver`,
/// and by the verifier as a `Component`.
pub struct LuminairComponents {
    add: Vec<AddComponent>,
}

impl LuminairComponents {
    /// Initializes components from claims and interaction elements.
    pub fn new(
        claims: &LuminairClaim,
        interaction_elements: &LuminairInteractionElements,
        interaction_claim: &LuminairInteractionClaim,
        is_first_log_sizes: &[u32],
    ) -> Self {
        let tree_span_provider = &mut TraceLocationAllocator::new_with_preproccessed_columns(
            &is_first_log_sizes
                .iter()
                .copied()
                .map(|log_size| IsFirst::new(log_size).id())
                .collect::<Vec<_>>(),
        );

        let add_components = claims
            .add
            .iter()
            .zip(interaction_claim.add.iter())
            .map(|(cl, int_cl)| {
                AddComponent::new(
                    tree_span_provider,
                    AddEval::new(cl, interaction_elements.node_lookup_elements.clone()),
                    int_cl.claimed_sum,
                )
            })
            .collect();

        Self {
            add: add_components,
        }
    }

    /// Returns the `ComponentProver` of each components, used by the prover.
    pub fn provers(&self) -> Vec<&dyn ComponentProver<SimdBackend>> {
        self.add
            .iter()
            .map(|c| c as &dyn ComponentProver<SimdBackend>)
            .collect()
    }

    /// Returns the `Component` of each components used by the verifier.
    pub fn components(&self) -> Vec<&dyn Component> {
        self.add.iter().map(|c| c as &dyn Component).collect()
    }
}
