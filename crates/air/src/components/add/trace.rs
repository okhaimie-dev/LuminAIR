use num_traits::identities::Zero;
use serde::{Deserialize, Serialize};
use stwo_prover::constraint_framework::logup::LookupElements;
use stwo_prover::core::backend::simd::m31::PackedBaseField;
use stwo_prover::core::backend::Column;
use stwo_prover::core::channel::Channel;
use stwo_prover::core::{
    backend::{simd::SimdBackend, Col},
    fields::m31::BaseField,
    poly::circle::{CanonicCoset, CircleEvaluation},
};

use crate::components::{AddClaim, TraceColumn, TraceEval};

/// Generate trace for element-wise addition of two vectors.
pub fn gen_add_trace(
    log_size: u32,
    lhs: &[PackedBaseField],
    rhs: &[PackedBaseField],
) -> (TraceEval, AddClaim, Vec<PackedBaseField>) {
    // Calculate trace size and initialize columns
    let trace_size = 1 << log_size;
    let mut trace = Vec::with_capacity(3);
    for _ in 0..3 {
        trace.push(Col::<SimdBackend, BaseField>::zeros(trace_size));
    }

    // Calculate actual size needed
    let size = lhs.len().max(rhs.len());

    // Prepare output data
    let mut c_data = Vec::with_capacity(size);

    // Fill trace and generate output data
    for i in 0..trace_size {
        if i < size {
            // Get values with broadcasting
            let lhs = lhs[i % lhs.len()];
            let rhs = rhs[i % rhs.len()];
            let out = lhs + rhs;

            trace[0].set(i, lhs.to_array()[0]);
            trace[1].set(i, rhs.to_array()[0]);
            trace[2].set(i, out.to_array()[0]);

            if i < size {
                c_data.push(out);
            }
        } else {
            // Pad remaining trace with zeros
            trace[0].set(i, BaseField::zero());
            trace[1].set(i, BaseField::zero());
            trace[2].set(i, BaseField::zero());
        }
    }

    // Create domain
    let domain = CanonicCoset::new(log_size).circle_domain();

    (
        trace
            .into_iter()
            .map(|eval| CircleEvaluation::new(domain, eval))
            .collect(),
        AddClaim {
            log_size,
            _marker: std::marker::PhantomData,
        },
        c_data,
    )
}

/// Enum representing the column indices in the Add trace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AddColumn {
    /// Index of the `lhs` register column in the Add trace.
    Lhs,
    /// Index of the `rhs` register column in the Add trace.
    Rhs,
    /// Index of the `res` register column in the Add trace.
    Res,
}

impl TraceColumn for AddColumn {
    fn count() -> (usize, usize) {
        (3, 1)
    }
}

/// The number of random elements necessary for the Add lookup argument.
const ADD_LOOKUP_ELEMENTS: usize = 3;

/// The interaction elements are drawn for the extension column of the Add component.
///
/// The logUp protocol uses these elements to combine the values of the different
/// registers of the main trace to create a random linear combination
/// of them, and use it in the denominator of the fractions in the logUp protocol.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AddElements(LookupElements<ADD_LOOKUP_ELEMENTS>);

impl AddElements {
    /// Provides dummy lookup elements.
    pub fn dummy() -> Self {
        Self(LookupElements::dummy())
    }

    /// Draw random elements from the Fiat-Shamir [`Channel`].
    ///
    /// These elements are randomly secured, and will be use
    /// to generate the interaction trace with the logUp protocol.
    pub fn draw(channel: &mut impl Channel) -> Self {
        Self(LookupElements::draw(channel))
    }
}
