use num_traits::{One, Zero};
use numerair::FixedPoint;
use serde::{Deserialize, Serialize};
use stwo_prover::{
    constraint_framework::{logup::LogupTraceGenerator, Relation},
    core::{
        backend::{
            simd::{column::BaseColumn, m31::LOG_N_LANES, qm31::PackedSecureField},
            Column,
        },
        fields::m31::BaseField,
        poly::circle::{CanonicCoset, CircleEvaluation},
    },
};

use crate::{
    components::{AddClaim, InteractionClaim, NodeElements, TraceColumn, TraceError, TraceEval},
    pie::NodeInfo,
};

/// Represents a single row of the [`AddTable`]
#[derive(Debug, Default, PartialEq, Eq, Clone)]
pub struct AddTableRow {
    lhs: BaseField,
    rhs: BaseField,
}

/// Generates the main trace for element-wise addition of two vectors.
pub fn trace_evaluation(
    lhs: &[BaseField],
    rhs: &[BaseField],
    node_info: &NodeInfo,
) -> (TraceEval, AddClaim, Vec<BaseField>) {
    let n_rows = lhs.len().min(rhs.len()) as u32;
    let log_n_rows = n_rows.next_power_of_two().ilog2();
    let log_size = log_n_rows + LOG_N_LANES;
    let mut trace = vec![BaseColumn::zeros(1 << log_size); AddColumn::count().0];

    let mut out = Vec::with_capacity(n_rows as usize);

    for i in 0..n_rows as usize {
        let lhs_val = lhs[i];
        let rhs_val = rhs[i];
        let out_val = lhs_val.fixed_add(rhs_val);

        trace[AddColumn::Lhs.index()].data[i] = lhs_val.into();
        trace[AddColumn::Rhs.index()].data[i] = rhs_val.into();
        trace[AddColumn::Out.index()].data[i] = out_val.into();

        out.push(out_val);
    }

    let domain = CanonicCoset::new(log_size).circle_domain();
    let trace = trace
        .into_iter()
        .map(|col| CircleEvaluation::new(domain, col))
        .collect();

    (trace, AddClaim::new(log_size, node_info.clone()), out)
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
