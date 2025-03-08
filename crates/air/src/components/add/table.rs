use serde::{Deserialize, Serialize};
use stwo_prover::core::{
    backend::{simd::column::BaseColumn, Column},
    fields::m31::BaseField,
    poly::circle::{CanonicCoset, CircleEvaluation},
};

use crate::{
    components::{AddClaim, TraceColumn, TraceError, TraceEval},
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

        // Fill columns
        for (vec_row, row) in self.table.iter().enumerate() {
            lhs_col.set(vec_row, row.lhs);
            rhs_col.set(vec_row, row.rhs);
            out_col.set(vec_row, row.out);
        }

        // Create domain
        let domain = CanonicCoset::new(log_size).circle_domain();

        // Create trace
        let mut trace = Vec::with_capacity(AddColumn::count().0);
        trace.push(CircleEvaluation::new(domain, lhs_col));
        trace.push(CircleEvaluation::new(domain, rhs_col));
        trace.push(CircleEvaluation::new(domain, out_col));

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
        (3, 0)
    }
}
