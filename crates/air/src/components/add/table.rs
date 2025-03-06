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
    components::{AddClaim, InteractionClaim, NodeElements, TraceColumn, TraceError, TraceEval},
    pie::NodeInfo,
    utils::{calculate_log_size, get_index},
};

/// Generates the main trace for element-wise addition of two tensors.
pub fn trace_evaluation(
    lhs: &[Fixed],
    rhs: &[Fixed],
    lexpr: &(Expression, Expression),
    rexpr: &(Expression, Expression),
    stack: &mut Vec<i64>,
    out_data: &mut Vec<Fixed>,
    node_info: &NodeInfo,
) -> (TraceEval, AddClaim) {
    // Calculate log size
    let log_size = calculate_log_size(out_data.len());

    // Calculate trace size
    let trace_size = 1 << log_size;

    // Create domain
    let domain = CanonicCoset::new(log_size).circle_domain();

    // Instantiate trace
    let mut trace = Vec::with_capacity(AddColumn::count().0);

    // Create columns
    let mut lhs_column = Col::<SimdBackend, BaseField>::zeros(trace_size);
    let mut rhs_column = Col::<SimdBackend, BaseField>::zeros(trace_size);
    let mut out_column = Col::<SimdBackend, BaseField>::zeros(trace_size);

    // Fill columns
    for i in 0..trace_size {
        if i < out_data.len() {
            let lhs_val = get_index(lhs, lexpr, stack, i);
            let rhs_val = get_index(rhs, rexpr, stack, i);
            let out_val = lhs_val + rhs_val;

            lhs_column.set(i, lhs_val.to_m31());
            rhs_column.set(i, rhs_val.to_m31());
            out_column.set(i, out_val.to_m31());

            out_data[i] = out_val
        } else {
            break;
        }
    }

    // Add columns to the trace
    trace.push(CircleEvaluation::new(domain.clone(), lhs_column));
    trace.push(CircleEvaluation::new(domain.clone(), rhs_column));
    trace.push(CircleEvaluation::new(domain, out_column));

    (trace, AddClaim::new(log_size, node_info.clone()))
}

/// Enum representing the column indices in the Add trace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AddColumn {
    /// Index of the `lhs` register column in the Add trace.
    Lhs,

    /// Index of the `rhs` register column in the Add trace.
    Rhs,

    /// Index of the `out` register column in the Add trace.
    Out,
}

impl AddColumn {
    /// Returns the index of the column in the Add trace.
    pub const fn index(self) -> usize {
        match self {
            Self::Lhs => 0,
            Self::Rhs => 1,
            Self::Out => 2,
        }
    }
}

impl TraceColumn for AddColumn {
    /// Returns the number of columns in the main trace and interaction trace.
    ///     
    /// For the Add component, both the main trace and interaction trace have 3 columns each.
    fn count() -> (usize, usize) {
        (3, 3)
    }
}

/// Generates the interaction trace for the Add component using the main trace and lookup elements.
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

    // Create trace for LHS
    let lhs_col = &main_trace_eval[AddColumn::Lhs.index()].data;
    let mut col_lhs = logup_gen.new_col();
    for row in 0..1 << (log_size - LOG_N_LANES) {
        let lhs = lhs_col[row];
        let multiplicity = if node_info.inputs[0].is_initializer {
            PackedSecureField::zero()
        } else {
            -PackedSecureField::one()
        };

        col_lhs.write_frac(row, multiplicity, lookup_elements.combine(&[lhs]));
    }
    col_lhs.finalize_col();

    // Create trace for RHS
    let rhs_col = &main_trace_eval[AddColumn::Rhs.index()].data;
    let mut col_rhs = logup_gen.new_col();
    for row in 0..1 << (log_size - LOG_N_LANES) {
        let rhs = rhs_col[row];
        let multiplicity = if node_info.inputs[1].is_initializer {
            PackedSecureField::zero()
        } else {
            -PackedSecureField::one()
        };

        col_rhs.write_frac(row, multiplicity, lookup_elements.combine(&[rhs]));
    }
    col_rhs.finalize_col();

    // Create trace for output
    let out_col = &main_trace_eval[AddColumn::Out.index()].data;
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
