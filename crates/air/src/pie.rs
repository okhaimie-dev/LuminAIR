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
    pub io_info: IOInfo,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ExecutionResources {
    pub op_counter: OpCounter,
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct OpCounter {
    pub add: Option<usize>,
}

/// Struct to hold input source information
///
/// is_initializer is true if the input is coming from an initial input.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InputInfo {
    pub is_initializer: bool,
}

/// Struct to hold output destination information
///
/// is_final_output is true if the output is a final output of the graph,
/// false if it's an intermediate output used by other nodes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OutputInfo {
    pub is_final_output: bool,
}

/// Struct to hold input/output information
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IOInfo {
    pub inputs: Vec<InputInfo>,
    pub output: OutputInfo,
}
