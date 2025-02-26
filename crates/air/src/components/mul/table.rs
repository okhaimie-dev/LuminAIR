use crate::{
    components::{InteractionClaim, MulClaim, NodeElements, TraceColumn, TraceError, TraceEval},
    pie::NodeInfo,
    utils::calculate_log_size,
};
use num_traits::{One, Zero};
use numerair::packed::FixedPackedBaseField;
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

/// Generates the main trace for element-wise multiplication of two tensors.
pub fn trace_evaluation(
    lhs: &[FixedPackedBaseField],
    rhs: &[FixedPackedBaseField],
    node_info: &NodeInfo,
) -> (TraceEval, MulClaim, Vec<FixedPackedBaseField>) {
    // Calculate actual size needed
    let actual_size = lhs.len().max(rhs.len());

    // Calculate log size
    let log_size = calculate_log_size(actual_size);

    // Calculate trace size
    let trace_size = 1 << log_size;

    // Create domain
    let domain = CanonicCoset::new(log_size).circle_domain();

    // Instantiate trace
    let mut trace = Vec::with_capacity(MulColumn::count().0);

    // Instantiate output vector
    let mut output_vec = Vec::with_capacity(actual_size);

    // Fill columns
    for column_idx in 0..MulColumn::count().0 {
        let mut column = Col::<SimdBackend, BaseField>::zeros(trace_size);

        for i in 0..trace_size {
            if i < actual_size {
                let lhs_val = lhs[i % lhs.len()];
                let rhs_val = rhs[i % rhs.len()];
                let (out_val, rem_val) = lhs_val * rhs_val;

                match column_idx {
                    0 => column.set(i, lhs_val.0.to_array()[0]),
                    1 => column.set(i, rhs_val.0.to_array()[0]),
                    2 => {
                        column.set(i, out_val.0.to_array()[0]);
                        output_vec.push(out_val);
                    }
                    3 => column.set(i, rem_val.0.to_array()[0]),
                    _ => unreachable!(),
                }
            }
        }

        trace.push(CircleEvaluation::new(domain, column));
    }

    (
        trace,
        MulClaim::new(log_size, node_info.clone()),
        output_vec,
    )
}

/// Enum representing the column indices in the Mul trace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MulColumn {
    /// Index of the `lhs` register column in the Mul trace.
    Lhs,

    /// Index of the `rhs` register column in the Mul trace.
    Rhs,

    /// Index of the `out` register column in the Mul trace.
    Out,

    /// Index of the `rem` register column in the Mul trace.
    Rem,
}

impl MulColumn {
    /// Returns the index of the column in the Mul trace.
    pub const fn index(self) -> usize {
        match self {
            Self::Lhs => 0,
            Self::Rhs => 1,
            Self::Out => 2,
            Self::Rem => 3,
        }
    }
}

impl TraceColumn for MulColumn {
    /// Returns the number of columns in the main trace and interaction trace.
    ///     
    /// For the Mul component, both the main trace and interaction trace have 3 columns each.
    fn count() -> (usize, usize) {
        (4, 3)
    }
}

/// Generates the interaction trace for the Mul component using the main trace and lookup elements.
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
    let lhs_col = &main_trace_eval[MulColumn::Lhs.index()].data;
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
    let rhs_col = &main_trace_eval[MulColumn::Rhs.index()].data;
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
    let out_col = &main_trace_eval[MulColumn::Out.index()].data;
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
