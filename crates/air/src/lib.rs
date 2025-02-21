#![feature(trait_upcasting)]

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

/// The STARK proof of the execution of a given Luminair Graph.
///
/// It includes the proof as well as the claims during the various phases of the proof generation.
#[derive(Serialize, Deserialize, Debug)]
pub struct LuminairProof<H: MerkleHasher> {
    pub claim: LuminairClaim,
    pub interaction_claim: LuminairInteractionClaim,
    pub proof: StarkProof<H>,
    pub execution_resources: ExecutionResources,
}

/// A claim over the log sizes for each component of the system.
///
/// A component is made of three types of trace:
/// - Preprocessed Trace (Phase 0)
/// - Main Trace (Phase 1)
/// - Interaction Trace (Phase 2)
#[derive(Serialize, Deserialize, Debug)]
pub struct LuminairClaim {
    pub add: Vec<AddClaim>,
    pub is_first_log_sizes: Vec<u32>,
}

impl LuminairClaim {
    pub fn init(is_first_log_sizes: Vec<u32>) -> Self {
        Self {
            add: vec![],
            is_first_log_sizes,
        }
    }

    pub fn mix_into(&self, channel: &mut impl Channel) {
        self.add.iter().for_each(|c| c.mix_into(channel));
    }

    pub fn log_sizes(&self) -> TreeVec<Vec<u32>> {
        let mut log_sizes = TreeVec::concat_cols(
            self.add
                .iter()
                .map(|c| c.log_sizes())
                .collect::<Vec<_>>()
                .into_iter(),
        );

        // We overwrite the preprocessed column claim to have all possible log sizes
        // in the merkle root for the verification.
        log_sizes[PREPROCESSED_TRACE_IDX] = self.is_first_log_sizes.clone();

        log_sizes
    }
}

/// A claim over the claimed sum of the interaction columns for each component of the system
///
/// Needed for the lookup protocol (logUp with AIR).
#[derive(Serialize, Deserialize, Debug)]
pub struct LuminairInteractionClaim {
    pub add: Vec<InteractionClaim>,
}

impl LuminairInteractionClaim {
    pub fn init() -> Self {
        Self { add: vec![] }
    }

    pub fn mix_into(&self, channel: &mut impl Channel) {
        self.add.iter().for_each(|c| c.mix_into(channel));
    }
}