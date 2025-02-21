use serde::{Deserialize, Serialize};
use stwo_prover::core::{
    backend::{Backend, Col, Column},
    fields::m31::BaseField,
    poly::{
        circle::{CanonicCoset, CircleEvaluation},
        BitReversedOrder,
    },
    ColumnVec,
};

/// Serializable form of a single trace evaluation, capturing its domain size and values.
#[derive(Clone, Serialize, Deserialize, Debug)]
struct SerializableCircleEvaluation {
    log_size: u32,
    values: Vec<BaseField>,
}

impl<B: Backend> From<&CircleEvaluation<B, BaseField, BitReversedOrder>>
    for SerializableCircleEvaluation
{
    /// Converts a `CircleEvaluation` into its serializable counterpart by extracting the log size and values.
    fn from(eval: &CircleEvaluation<B, BaseField, BitReversedOrder>) -> Self {
        Self {
            log_size: eval.domain.log_size(),
            values: eval.values.to_cpu(),
        }
    }
}

/// Serializable collection of trace evaluations, used for storing or transmitting multiple trace columns.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct SerializableTrace {
    evaluations: Vec<SerializableCircleEvaluation>,
}

impl SerializableTrace {
    /// Reconstructs the original trace columns from the serializable data, creating a `ColumnVec` of `CircleEvaluation`s.
    pub fn to_trace<B: Backend>(
        &self,
    ) -> ColumnVec<CircleEvaluation<B, BaseField, BitReversedOrder>> {
        self.evaluations
            .iter()
            .map(|eval| {
                let domain = CanonicCoset::new(eval.log_size).circle_domain();
                // Create a new column with the values
                let mut col = Col::<B, BaseField>::zeros(eval.values.len());
                for (i, val) in eval.values.iter().enumerate() {
                    col.set(i, *val);
                }
                CircleEvaluation::new(domain, col)
            })
            .collect()
    }
}

impl<B: Backend> From<&ColumnVec<CircleEvaluation<B, BaseField, BitReversedOrder>>>
    for SerializableTrace
{
    /// Converts a collection of trace evaluations into a serializable format by mapping each `CircleEvaluation` to its serializable form.
    fn from(trace: &ColumnVec<CircleEvaluation<B, BaseField, BitReversedOrder>>) -> Self {
        Self {
            evaluations: trace
                .iter()
                .map(SerializableCircleEvaluation::from)
                .collect(),
        }
    }
}
