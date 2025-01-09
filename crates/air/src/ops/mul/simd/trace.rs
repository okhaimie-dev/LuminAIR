use crate::ops::mul::SCALE_FACTOR_INV;
use crate::tensor::AirTensor;
use num_traits::identities::Zero;
use stwo_prover::core::backend::Column;
use stwo_prover::core::{
    backend::{
        simd::{m31::PackedBaseField, SimdBackend},
        Col,
    },
    fields::m31::BaseField,
    poly::{
        circle::{CanonicCoset, CircleEvaluation},
        BitReversedOrder,
    },
    ColumnVec,
};

pub(super) fn generate_trace<'a>(
    log_size: u32,
    a: &'a AirTensor<'a, PackedBaseField>,
    b: &'a AirTensor<'a, PackedBaseField>,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    AirTensor<'static, PackedBaseField>,
) {
    // Calculate trace size and initialize columns
    let trace_size = 1 << log_size;
    let mut trace = Vec::with_capacity(3);
    for _ in 0..3 {
        trace.push(Col::<SimdBackend, BaseField>::zeros(trace_size));
    }

    // Calculate actual size needed
    let size = a.size().max(b.size());

    // Prepare output data
    let mut c_data = Vec::with_capacity(size);

    // Fill trace and generate output data
    for i in 0..trace_size {
        if i < size {
            // Get values with broadcasting
            let a_val = a.data()[i % a.data().len()];
            let b_val = b.data()[i % b.data().len()];

            // Store original values in trace
            trace[0].set(i, a_val.to_array()[0]);
            trace[1].set(i, b_val.to_array()[0]);

            // For debugging
            println!("i: {}", i);
            println!("a_val raw: {:?}", a_val);
            println!("b_val raw: {:?}", b_val);


            // First multiply values (fixed point multiplication)
            let mut product = a_val * b_val;
            println!("Product before scaling: {:?}", product);

            // Apply scaling compensation (divide by scale factor)
            // Since values are already scaled up by 2^20, the product is scaled by 2^40
            // We need to divide by 2^20 to get back to 2^20 scaling
            let scale_compensated = product * PackedBaseField::broadcast(*SCALE_FACTOR_INV);
            println!("Product after scaling: {:?}", scale_compensated);

            trace[2].set(i, scale_compensated.to_array()[0]);

            if i < size {
                c_data.push(scale_compensated);
            }
        } else {
            // Pad remaining trace with zeros
            trace[0].set(i, BaseField::zero());
            trace[1].set(i, BaseField::zero());
            trace[2].set(i, BaseField::zero());
        }
    }

    // Create output tensor
    let c = AirTensor::Owned {
        data: c_data,
        dims: if a.size() > b.size() {
            a.dims().to_vec()
        } else {
            b.dims().to_vec()
        },
    };

    let domain = CanonicCoset::new(log_size).circle_domain();

    (
        trace
            .into_iter()
            .map(|eval| CircleEvaluation::new(domain, eval))
            .collect(),
        c,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use numerair::fixed_points::{pack_floats, unpack_floats, DEFAULT_SCALE};
    use stwo_prover::core::backend::simd::m31::LOG_N_LANES;

    #[test]
    fn test_generate_trace_correctness() {
        let binding_a_1 = pack_floats(&[1.0; 4], DEFAULT_SCALE, 0.0);
        let binding_b_1 = pack_floats(&[2.0; 4], DEFAULT_SCALE, 0.0);
        let binding_a_2 = pack_floats(&[5.0; 1], DEFAULT_SCALE, 0.0);
        let binding_b_2 = pack_floats(&[1.0; 6], DEFAULT_SCALE, 0.0);
        let binding_a_3 = pack_floats(&[1.0; 3], DEFAULT_SCALE, 0.0);
        let binding_b_3 = pack_floats(&[2.0; 6], DEFAULT_SCALE, 0.0);
        let binding_a_4 = pack_floats(&[0.4, 0.1, 3.3, 4.0], DEFAULT_SCALE, 0.0);
        let binding_b_4 = pack_floats(&[0.3, 0.2, 7.3, 8.0], DEFAULT_SCALE, 0.0);

        let test_cases = vec![
            // // Case 1: Simple 2x2 matrices
            // (
            //     AirTensor::new(&binding_a_1, vec![2, 2]),
            //     AirTensor::new(&binding_b_1, vec![2, 2]),
            //     vec![2, 2],
            //     pack_floats(&[2.0; 4], DEFAULT_SCALE, 0.0), // Expected result: 1 * 2 = 2 for all elements
            // ),
            // // Case 2: Broadcasting scalar to matrix
            // (
            //     AirTensor::new(&binding_a_2, vec![1]),
            //     AirTensor::new(&binding_b_2, vec![2, 3]),
            //     vec![2, 3],
            //     pack_floats(&[5.0; 6], DEFAULT_SCALE, 0.0), // Expected result: 5 * 1 = 5 for all elements
            // ),
            // // Case 3: Broadcasting row to matrix
            // (
            //     AirTensor::new(&binding_a_3, vec![1, 3]),
            //     AirTensor::new(&binding_b_3, vec![2, 3]),
            //     vec![2, 3],
            //     pack_floats(&[2.0; 6], DEFAULT_SCALE, 0.0), // Expected result: 1 * 2 = 2 for all elements
            // ),
            // Case 4: Different values in matrices
            (
                AirTensor::new(&binding_a_4, vec![2, 2]),
                AirTensor::new(&binding_b_4, vec![2, 2]),
                vec![2, 2],
                pack_floats(
                    &[0.4 * 0.3, 0.1 * 0.2, 3.3 * 7.3, 4.0 * 8.0],
                    DEFAULT_SCALE,
                    0.0,
                ), // Element-wise multiplication
            ),
        ];

        for (i, (tensor_a, tensor_b, expected_dims, expected_values)) in
            test_cases.into_iter().enumerate()
        {
            // Calculate required log_size
            let max_size = tensor_a.size().max(tensor_b.size());
            let required_log_size = ((max_size + (1 << LOG_N_LANES) - 1) >> LOG_N_LANES)
                .next_power_of_two()
                .trailing_zeros()
                + LOG_N_LANES;

            // Generate trace and get result tensor
            let (trace, result) = generate_trace(required_log_size, &tensor_a, &tensor_b);

            // Check trace size
            assert_eq!(
                trace.len(),
                3,
                "Case {}: Trace should have 3 columns (a, b, c)",
                i
            );
            assert_eq!(
                trace[0].len(),
                1 << required_log_size,
                "Case {}: Trace size should be 2^{}",
                i,
                required_log_size
            );

            // Verify result tensor dimensions
            assert_eq!(
                result.dims(),
                expected_dims,
                "Case {}: Result tensor has incorrect dimensions",
                i
            );

            // Unpack and verify result values
            let result_values = unpack_floats(result.data(), DEFAULT_SCALE, 0.0, result.size());
            let expected_values = unpack_floats(
                &expected_values,
                DEFAULT_SCALE,
                0.0,
                expected_dims.iter().product(),
            );

            assert_eq!(
                result_values,
                expected_values,
                "Case {}: Result tensor has incorrect values",
                i + 1
            );
        }
    }
}
