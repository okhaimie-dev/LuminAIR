use add::{
    component::{AddComponent, AddEval},
    table::{AddColumn, AddElements},
};
use serde::{Deserialize, Serialize};
use stwo_prover::{
    constraint_framework::{preprocessed_columns::PreprocessedColumn, TraceLocationAllocator},
    core::{
        air::{Component, ComponentProver},
        backend::simd::SimdBackend,
        channel::Channel,
        fields::{m31::BaseField, qm31::SecureField, secure_column::SECURE_EXTENSION_DEGREE},
        pcs::TreeVec,
        poly::{circle::CircleEvaluation, BitReversedOrder},
        ColumnVec,
    },
};
use thiserror::Error;

use crate::{
    pie::{IOInfo, OpCounter},
    LuminairClaim, LuminairInteractionClaim,
};

pub mod add;

/// Custom error type for the Trace.
#[derive(Debug, Error, Eq, PartialEq)]
pub enum TraceError {
    /// The component trace is empty.
    #[error("The trace is empty.")]
    EmptyTrace,
}

/// Type for trace evaluation to be used in Stwo.
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
    /// Logarithmic size (`log2`) of the evaluated trace.
    pub log_size: u32,
    /// Inputs/output information.
    pub io_info: IOInfo,
    /// Marker for the trace type.
    _marker: std::marker::PhantomData<T>,
}

impl<T: TraceColumn> Claim<T> {
    /// Creates a new claim.
    pub const fn new(log_size: u32, io_info: IOInfo) -> Self {
        Self {
            log_size,
            io_info,
            _marker: std::marker::PhantomData,
        }
    }

    /// Returns the `log_size` for each type of trace committed for the given trace type:
    /// - Preprocessed trace,
    /// - Main trace,
    /// - Interaction trace.
    ///
    /// The number of columns of each trace is known before actually evaluating them.
    /// The `log_size` is known once the main trace has been evaluated, to which we add
    /// [`stwo_prover::core::backend::simd::m31::LOG_N_LANES`]
    /// for the [`stwo_prover::core::backend::simd::SimdBackend`])
    ///
    /// Each element of the [`TreeVec`] is dedicated to the commitment of one type of trace.
    /// First element is for the preprocessed trace, second for the main trace and third for the
    /// interaction one.
    ///
    /// NOTE: Currently only the main trace is provided.
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

    /// Mix the log size of the table to the Fiat-Shamir [`Channel`],
    /// to bound the channel randomness and the trace.
    pub fn mix_into(&self, channel: &mut impl Channel) {
        channel.mix_u64(self.log_size.into());
    }
}

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

/// All the interaction elements required by the components during the interaction phas 2.
///
/// The elements are drawn from a Fiat-Shamir [`Channel`], currently using the BLAKE2 hash.
#[derive(Clone, Debug)]
pub struct LuminairInteractionElements {
    pub add_lookup_elements: AddElements,
}

impl LuminairInteractionElements {
    /// Draw all the interaction elements needed for
    /// all the components of the system.
    pub fn draw(channel: &mut impl Channel, op_counter: &OpCounter) -> Self {
        // Only draw elements once and reuse them
        let add_elements = if op_counter.add.unwrap_or(0) > 0 {
            AddElements::draw(channel)
        } else {
            AddElements::dummy()
        };

        Self {
            add_lookup_elements: add_elements,
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
    /// Initilizes all the LuminAIR components from the claims generated from the trace.
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
                .map(PreprocessedColumn::IsFirst)
                .collect::<Vec<_>>(),
        );

        let add_components = claims
            .add
            .iter()
            .zip(interaction_claim.add.iter())
            .map(|(cl, int_cl)| {
                AddComponent::new(
                    tree_span_provider,
                    AddEval::new(cl, interaction_elements.add_lookup_elements.clone()),
                    (int_cl.claimed_sum, None),
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
