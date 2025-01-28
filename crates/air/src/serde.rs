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

/// Serializable representation of a trace evaluation
#[derive(Serialize, Deserialize)]
pub struct SerializableCircleEvaluation {
    pub log_size: u32,
    pub values: Vec<BaseField>,
}

impl<B: Backend> From<&CircleEvaluation<B, BaseField, BitReversedOrder>>
    for SerializableCircleEvaluation
{
    fn from(eval: &CircleEvaluation<B, BaseField, BitReversedOrder>) -> Self {
        Self {
            log_size: eval.domain.log_size(),
            values: eval.values.to_cpu(),
        }
    }
}

/// Serializable representation of a trace
#[derive(Serialize, Deserialize)]
pub struct SerializableTrace {
    pub evaluations: Vec<SerializableCircleEvaluation>,
}

impl SerializableTrace {
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
    fn from(trace: &ColumnVec<CircleEvaluation<B, BaseField, BitReversedOrder>>) -> Self {
        Self {
            evaluations: trace
                .iter()
                .map(SerializableCircleEvaluation::from)
                .collect(),
        }
    }
}
