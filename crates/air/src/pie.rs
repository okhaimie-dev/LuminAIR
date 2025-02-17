use serde::{Deserialize, Serialize};

use crate::{components::ClaimType, serde::SerializableTrace};

#[derive(Serialize, Deserialize, Debug)]
pub struct LuminairPie {
    pub traces: Vec<Trace>,
    pub execution_resources: ExecutionResources,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Trace {
    pub eval: SerializableTrace,
    pub claim: ClaimType,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ExecutionResources {
    pub op_counter: OpCounter,
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct OpCounter {
    pub add: Option<usize>,
}
