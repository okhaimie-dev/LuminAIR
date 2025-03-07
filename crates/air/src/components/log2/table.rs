use luminal::shape::Expression;
use num_traits::{One, Zero};
use numerair::Fixed;
use serde::{Deserialize, Serialize};
use stwo_prover::{
    constraint_framework::{logup::LogupTraceGenerator, Relation},
    core::{
        backend::{
            simd::{m31::LOG_N_LANES, qm31::PackedSecureField, SimdBackend},
            Col, Column,
        },
        fields::m31::BaseField,
        poly::circle::{CanonicCoset, CircleEvaluation},
    },
};

use crate::{
    components::{InteractionClaim, Log2Claim, NodeElements, TraceColumn, TraceError, TraceEval},
    pie::NodeInfo,
    utils::{calculate_log_size, get_index},
};

/// Generates the main trace for element-wise log2 of a tensor.
pub fn trace_evaluation(
    input: &[Fixed],
    expr: &(Expression, Expression),
    stack: &mut Vec<i64>,
    out_data: &mut Vec<Fixed>,
    node_info: &NodeInfo,
) -> (TraceEval, Log2Claim) {
    let log_size = calculate_log_size(out_data.len());
    let trace_size = 1 << log_size;
    let domain = CanonicCoset::new(log_size).circle_domain();

    let mut trace = Vec::with_capacity(Log2Column::count().0);

    let mut input_column = Col::<SimdBackend, BaseField>::zeros(trace_size);
    let mut out_column = Col::<SimdBackend, BaseField>::zeros(trace_size);

    for i in 0..trace_size {
        if i < out_data.len() {
            let input_val = get_index(input, expr, stack, i);
            let out_val = input_val.log2(); // TODO: implement this

            input_column.set(i, input_val.to_m31());
            out_column.set(i, out_val.to_m31());

            out_data[i] = out_val;
        } else {
            break;
        }
    }

    trace.push(CircleEvaluation::new(domain.clone(), input_column));
    trace.push(CircleEvaluation::new(domain, out_column));

    (trace, Log2Claim::new(log_size, node_info.clone()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Log2Column {
    Input,
    Out,
}

impl Log2Column {
    pub const fn index(self) -> usize {
        match self {
            Self::Input => 0,
            Self::Out => 1,
        }
    }
}

impl TraceColumn for Log2Column {
    fn count() -> (usize, usize) {
        (2, 2) // Main trace and interaction trace both have 2 columns
    }
}

pub fn interaction_trace_evaluation(
    main_trace_eval: &TraceEval,
    lookup_elements: &NodeElements,
    node_info: &NodeInfo,
) -> Result<(TraceEval, InteractionClaim), TraceError> {
    if main_trace_eval.is_empty() {
        return Err(TraceError::EmptyTrace);
    }

    let log_size = main_trace_eval[0].domain.log_size();
    let mut logup_gen = LogupTraceGenerator::new(log_size);

    // Input trace
    let input_col = &main_trace_eval[Log2Column::Input.index()].data;
    let mut col_input = logup_gen.new_col();
    for row in 0..1 << (log_size - LOG_N_LANES) {
        let input = input_col[row];
        let multiplicity = if node_info.inputs[0].is_initializer {
            PackedSecureField::zero()
        } else {
            -PackedSecureField::one()
        };
        col_input.write_frac(row, multiplicity, lookup_elements.combine(&[input]));
    }
    col_input.finalize_col();

    // Output trace
    let out_col = &main_trace_eval[Log2Column::Out.index()].data;
    let mut col_out = logup_gen.new_col();
    for row in 0..1 << (log_size - LOG_N_LANES) {
        let out = out_col[row];
        let multiplicity = if node_info.output.is_final_output {
            PackedSecureField::zero()
        } else {
            PackedSecureField::one() * BaseField::from_u32_unchecked(node_info.num_consumers as u32)
        };
        col_out.write_frac(row, multiplicity, lookup_elements.combine(&[out]));
    }
    col_out.finalize_col();

    let (trace, claimed_sum) = logup_gen.finalize_last();
    Ok((trace, InteractionClaim { claimed_sum }))
}
