use num_traits::Zero;
use serde::{Deserialize, Serialize};
use stwo_prover::{
    constraint_framework::logup::LookupElements,
    core::{
        backend::{
            simd::{m31::PackedBaseField, SimdBackend},
            Col, Column,
        },
        channel::Channel,
        fields::m31::BaseField,
        poly::circle::{CanonicCoset, CircleEvaluation},
    },
};

use crate::components::{AddClaim, TraceColumn, TraceEval};

/// Generate trace for element-wise addition of two vectors.
pub fn trace_evaluation(
    log_size: u32,
    lhs: &[PackedBaseField],
    rhs: &[PackedBaseField],
    lhs_is_initializer: bool,
    rhs_is_initializer: bool,
) -> (TraceEval, AddClaim, Vec<PackedBaseField>) {
    // Calculate trace size
    let trace_size = 1 << log_size;
    // Calculate actual size needed
    let size = lhs.len().max(rhs.len());
    // Create domain
    let domain = CanonicCoset::new(log_size).circle_domain();

    // Initialize result vector
    let mut main_trace = Vec::with_capacity(3);

    // Prepare output data
    let mut output = Vec::with_capacity(size);

    // Create separate columns and fill them
    for column_idx in 0..3 {
        let mut column = Col::<SimdBackend, BaseField>::zeros(trace_size);

        for i in 0..trace_size {
            if i < size {
                // Get values with broadcasting
                let lhs_val = lhs[i % lhs.len()];
                let rhs_val = rhs[i % rhs.len()];
                let out_val = lhs_val + rhs_val;

                // Set appropriate value based on column index
                match column_idx {
                    0 => column.set(i, lhs_val.to_array()[0]),
                    1 => column.set(i, rhs_val.to_array()[0]),
                    2 => column.set(i, out_val.to_array()[0]),
                    _ => unreachable!(),
                }

                output.push(out_val)
            } else {
                // Pad with zeros
                column.set(i, BaseField::zero());
            }
        }

        main_trace.push(CircleEvaluation::new(domain, column));
    }

    (main_trace, AddClaim::new(log_size), output)
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
    fn count() -> (usize, usize) {
        (3, 1)
    }
}

/// The number of random elements necessary for the Add lookup argument.
const ADD_LOOKUP_ELEMENTS: usize = 1;

/// The interaction elements are drawn for the extension column of the Add component.
///
/// The logUp protocol uses these elements to combine the values of the different
/// registers of the main trace to create a random linear combination
/// of them, and use it in the denominator of the fraction in the logUp protocol.
///
/// There is 1 lookup element in the Add component: `out`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AddElements(LookupElements<ADD_LOOKUP_ELEMENTS>);

impl AddElements {
    /// Draw random elements from the Fiat-Shamir [`Channel`].
    ///
    /// These elements are randomly secured, and will be use
    /// to generate the interaction trace with the logUp protocol.
    pub fn draw(channel: &mut impl Channel) -> Self {
        Self(LookupElements::draw(channel))
    }
}
