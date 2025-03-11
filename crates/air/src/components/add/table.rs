use serde::{Deserialize, Serialize};
use stwo_prover::{
    constraint_framework::{logup::LogupTraceGenerator, Relation},
    core::{
        backend::{
            simd::{column::BaseColumn, m31::LOG_N_LANES},
            Column,
        },
        fields::m31::BaseField,
        poly::circle::{CanonicCoset, CircleEvaluation},
    },
};

use crate::{
    components::{AddClaim, InteractionClaim, NodeElements, TraceColumn, TraceError, TraceEval},
    utils::calculate_log_size,
};

/// Represents the trace for the Add component, containing the required registers for its
/// constraints.
#[derive(Debug, Default, PartialEq, Eq, Clone)]
pub struct AddTable {
    /// A vector of [`AddTableRow`] representing the table rows.
    pub table: Vec<AddTableRow>,
}

/// Represents a single row of the [`AddTable`]
#[derive(Debug, Default, PartialEq, Eq, Clone)]
pub struct AddTableRow {
    pub lhs: BaseField,
    pub rhs: BaseField,
    pub out: BaseField,
    pub lhs_mult: BaseField,
    pub rhs_mult: BaseField,
    pub out_mult: BaseField,
}

impl AddTable {
    /// Creates a new, empty [`AddTable`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a new row to the Add Table.
    pub fn add_row(&mut self, row: AddTableRow) {
        self.table.push(row);
    }

    /// Transforms the [`AddTable`] into [`TraceEval`] to be commited
    /// when generating a STARK proof.
    pub fn trace_evaluation(&self) -> Result<(TraceEval, AddClaim), TraceError> {
        let n_rows = self.table.len();
        if n_rows == 0 {
            return Err(TraceError::EmptyTrace);
        }
        // Calculate log size
        let log_size = calculate_log_size(n_rows);

        // Calculate trace size
        let trace_size = 1 << log_size;

        // Create columns
        let mut lhs_col = BaseColumn::zeros(trace_size);
        let mut rhs_col = BaseColumn::zeros(trace_size);
        let mut out_col = BaseColumn::zeros(trace_size);
        let mut lhs_mult_col = BaseColumn::zeros(trace_size);
        let mut rhs_mult_col = BaseColumn::zeros(trace_size);
        let mut out_mult_col = BaseColumn::zeros(trace_size);

        // Fill columns
        for (vec_row, row) in self.table.iter().enumerate() {
            lhs_col.set(vec_row, row.lhs);
            rhs_col.set(vec_row, row.rhs);
            out_col.set(vec_row, row.out);
            lhs_mult_col.set(vec_row, row.lhs_mult);
            rhs_mult_col.set(vec_row, row.rhs_mult);
            out_mult_col.set(vec_row, row.out_mult);
        }

        // Create domain
        let domain = CanonicCoset::new(log_size).circle_domain();

        // Create trace
        let mut trace = Vec::with_capacity(AddColumn::count().0);
        trace.push(CircleEvaluation::new(domain, lhs_col));
        trace.push(CircleEvaluation::new(domain, rhs_col));
        trace.push(CircleEvaluation::new(domain, out_col));
        trace.push(CircleEvaluation::new(domain, lhs_mult_col));
        trace.push(CircleEvaluation::new(domain, rhs_mult_col));
        trace.push(CircleEvaluation::new(domain, out_mult_col));

        Ok((trace, AddClaim::new(log_size)))
    }
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

    /// Index of the `lhs` multiplicity register column in the Add trace.
    LhsMult,

    /// Index of the `rhs` multiplicity register column in the Add trace.
    RhsMult,

    /// Index of the `out` multiplicity register column in the Add trace.
    OutMult,
}

impl AddColumn {
    /// Returns the index of the column in the Add trace.
    pub const fn index(self) -> usize {
        match self {
            Self::Lhs => 0,
            Self::Rhs => 1,
            Self::Out => 2,
            Self::LhsMult => 3,
            Self::RhsMult => 4,
            Self::OutMult => 5,
        }
    }
}
impl TraceColumn for AddColumn {
    /// Returns the number of columns in the main trace and interaction trace.
    fn count() -> (usize, usize) {
        (6, 3)
    }
}

/// Generates the interaction trace for the Add component using the main trace and lookup elements.
pub fn interaction_trace_evaluation(
    main_trace_eval: &TraceEval,
    lookup_elements: &NodeElements,
) -> Result<(TraceEval, InteractionClaim), TraceError> {
    if main_trace_eval.is_empty() {
        return Err(TraceError::EmptyTrace);
    }

    let log_size = main_trace_eval[0].domain.log_size();
    let mut logup_gen = LogupTraceGenerator::new(log_size);

    // Create trace for LHS
    let lhs_main_col = &main_trace_eval[AddColumn::Lhs.index()].data;
    let lhs_mult_col = &main_trace_eval[AddColumn::LhsMult.index()].data;
    let mut lhs_int_col = logup_gen.new_col();
    for row in 0..1 << (log_size - LOG_N_LANES) {
        let lhs = lhs_main_col[row];
        let multiplicity = lhs_mult_col[row];

        lhs_int_col.write_frac(row, multiplicity.into(), lookup_elements.combine(&[lhs]));
    }
    lhs_int_col.finalize_col();

    // Create trace for RHS
    let rhs_main_col = &main_trace_eval[AddColumn::Rhs.index()].data;
    let rhs_mult_col = &main_trace_eval[AddColumn::RhsMult.index()].data;
    let mut rhs_int_col = logup_gen.new_col();
    for row in 0..1 << (log_size - LOG_N_LANES) {
        let rhs = rhs_main_col[row];
        let multiplicity = rhs_mult_col[row];

        rhs_int_col.write_frac(row, multiplicity.into(), lookup_elements.combine(&[rhs]));
    }
    rhs_int_col.finalize_col();

    // Create trace for OUTPUT
    let out_main_col = &main_trace_eval[AddColumn::Out.index()].data;
    let out_mult_col = &main_trace_eval[AddColumn::OutMult.index()].data;
    let mut out_int_col = logup_gen.new_col();
    for row in 0..1 << (log_size - LOG_N_LANES) {
        let out = out_main_col[row];
        let multiplicity = out_mult_col[row];

        out_int_col.write_frac(row, multiplicity.into(), lookup_elements.combine(&[out]));
    }
    out_int_col.finalize_col();

    let (trace, claimed_sum) = logup_gen.finalize_last();

    Ok((trace, InteractionClaim { claimed_sum }))
}
