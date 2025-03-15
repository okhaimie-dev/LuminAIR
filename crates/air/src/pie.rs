use serde::{Deserialize, Serialize};

use crate::{
    components::{add::table::AddTable, mul::table::MulTable, AddClaim, ClaimType, MulClaim, TraceEval, TraceError},
    serde::SerializableTrace,
};

/// Represents an operator's trace table along with its claim before conversion
/// to a serialized trace format. Used to defer trace evaluation until proving.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum TableTrace {
    /// Addition operator trace table and claim.
    Add {
        table: AddTable,
        claim: AddClaim
    },
    /// Multiplication operator trace table and claim.
    Mul {
        table: MulTable,
        claim: MulClaim
    },
}

impl TableTrace {
    /// Creates a new [`TableTrace`] from an [`AddTable`] and a log size
    /// for use in the proof generation.
    pub fn from_add(table: AddTable, log_size: u32) -> Self {
        Self::Add {
            table,
            claim: AddClaim::new(log_size)
        }
    }
    
    /// Creates a new [`TableTrace`] from a [`MulTable`] and a log size
    /// for use in the proof generation.
    pub fn from_mul(table: MulTable, log_size: u32) -> Self {
        Self::Mul {
            table,
            claim: MulClaim::new(log_size)
        }
    }

    pub fn to_trace(&self) -> Result<(TraceEval, ClaimType), TraceError> {
        match self {
            TableTrace::Add { table, claim } => {
                let (trace, _) = table.trace_evaluation()?;
                Ok((trace, ClaimType::Add(claim.clone())))
            },

            TableTrace::Mul { table, claim } => {
                let (trace, _) = table.trace_evaluation()?;
                Ok((trace, ClaimType::Mul(claim.clone())))
            },
        }
    }
}

/// Container for traces and execution resources of a computational graph.
#[derive(Serialize, Deserialize, Debug)]
pub struct LuminairPie {
    pub traces: Vec<Trace>,
    pub table_traces: Vec<TableTrace>,
    pub execution_resources: ExecutionResources,
}

/// Represents a single trace with its evaluation, claim, and node information.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Trace {
    pub eval: SerializableTrace,
    pub claim: ClaimType,
}

impl Trace {
    pub fn new(eval: SerializableTrace, claim: ClaimType) -> Self {
        Self { eval, claim }
    }
}

/// Holds resource usage data for the execution.
#[derive(Serialize, Deserialize, Debug)]
pub struct ExecutionResources {
    pub op_counter: OpCounter,
    pub max_log_size: u32,
}

/// Counts occurrences of specific operations.
#[derive(Serialize, Deserialize, Debug, Default)]
pub struct OpCounter {
    pub add: Option<usize>,
    pub mul: Option<usize>,
}

/// Indicates if a node input is an initializer (i.e., from initial input).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InputInfo {
    pub is_initializer: bool,
    pub id: u32,
}

/// Indicates if a node output is a final graph output or intermediate.
#[derive(Debug, Clone, Serialize, Default, Deserialize, PartialEq, Eq)]
pub struct OutputInfo {
    pub is_final_output: bool,
}

/// Contains input, output, and consumer information for a node.
#[derive(Debug, Clone, Serialize, Default, Deserialize, PartialEq, Eq)]
pub struct NodeInfo {
    pub inputs: Vec<InputInfo>,
    pub output: OutputInfo,
    pub num_consumers: u32,
    pub id: u32,
}
