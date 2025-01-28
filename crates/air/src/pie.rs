use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use stwo_prover::core::{channel::Channel, pcs::TreeVec};

use crate::{
    components::{add::trace::AddColumn, Claim},
    serde::SerializableTrace,
};

#[derive(Serialize, Deserialize)]
pub struct LuminairPie {
    pub execution_resources: Option<ExecutionResources>,
    pub traces: HashMap<usize, Trace>,
}

#[derive(Serialize, Deserialize)]
pub struct ExecutionResources {
    op_counter: OpCounter,
}

#[derive(Serialize, Deserialize)]
enum OpCounter {
    Add(u32),
}

#[derive(Serialize, Deserialize)]
pub struct Trace {
    pub phase_1: Phase1,
    pub phase_2: Option<Phase2>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ClaimType {
    Add(Claim<AddColumn>),
}

impl ClaimType {
    pub fn mix_into(&self, channel: &mut impl Channel) {
        match self {
            ClaimType::Add(claim) => claim.mix_into(channel),
        }
    }

    pub fn log_size(&self) -> TreeVec<Vec<u32>> {
        match self {
            ClaimType::Add(claim) => claim.log_sizes(),
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct Phase1 {
    pub trace: SerializableTrace,
    pub claim: ClaimType,
}

#[derive(Serialize, Deserialize)]
pub struct Phase2 {
    pub trace: SerializableTrace,
    pub claim: ClaimType,
}
