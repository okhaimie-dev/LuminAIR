#![feature(trait_upcasting)]

use std::vec;

use ::serde::{Deserialize, Serialize};
use components::{AddClaim, InteractionClaim};
use pie::ExecutionResources;
use stwo_prover::constraint_framework::PREPROCESSED_TRACE_IDX;
use stwo_prover::core::{
    channel::Channel, pcs::TreeVec, prover::StarkProof, vcs::ops::MerkleHasher,
};

pub mod components;
pub mod pie;
pub mod serde;
pub mod utils;

/// STARK proof for a Luminair computational graph execution.
///
/// Contains the proof and claims from all proof generation phases.
#[derive(Serialize, Deserialize, Debug)]
pub struct LuminairProof<H: MerkleHasher> {
    pub claim: LuminairClaim,
    pub interaction_claim: LuminairInteractionClaim,
    pub proof: StarkProof<H>,
    pub execution_resources: ExecutionResources,
}

/// Claim for system components.
#[derive(Serialize, Deserialize, Debug)]
pub struct LuminairClaim {
    pub add: Option<AddClaim>,
    pub is_first_log_sizes: Vec<u32>,
}

impl LuminairClaim {
    /// Initializes a new claim with specified preprocessed trace log sizes.
    pub fn new(is_first_log_sizes: Vec<u32>) -> Self {
        Self {
            add: None,
            is_first_log_sizes,
        }
    }

    /// Mixes claim data into a Fiat-Shamir channel for proof binding.
    pub fn mix_into(&self, channel: &mut impl Channel) {
        if let Some(ref add) = self.add {
            add.mix_into(channel);
        }
    }

    /// Computes log sizes for all trace types in the claim.
    pub fn log_sizes(&self) -> TreeVec<Vec<u32>> {
        let mut log_sizes = TreeVec::new(vec![]);

        if let Some(ref add) = self.add {
            log_sizes = TreeVec::concat_cols([add.log_sizes()].into_iter());
        }

        log_sizes[PREPROCESSED_TRACE_IDX] = self.is_first_log_sizes.clone();
        log_sizes
    }
}

/// Claim over the sum of interaction columns per system component.
///
/// Used in the logUp lookup protocol with AIR.
#[derive(Serialize, Deserialize, Default, Debug)]
pub struct LuminairInteractionClaim {
    pub add: Option<InteractionClaim>,
}

impl LuminairInteractionClaim {
    /// Mixes interaction claim data into a Fiat-Shamir channel.
    pub fn mix_into(&self, channel: &mut impl Channel) {
        if let Some(ref add) = self.add {
            add.mix_into(channel);
        }
    }
}
